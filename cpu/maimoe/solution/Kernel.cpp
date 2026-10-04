#pragma GCC target("cpu=tsv110")
#include "maimoe/kernel_api.hpp"
#include "MaiMoeEngine.hpp"
#include "maimoe/render.hpp"
#include "maimoe/state.hpp"
#include "maimoe/sha256.hpp"
#include <cstring>
#include <arm_neon.h>
#include <bit>
#include <fstream>
#include <type_traits>
#include "maimoe/simai.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <limits>
#include <numeric>
#include <sstream>
#include <tuple>
#include <utility>

namespace maimoe::fast {
// Compact kernel-private events. The public base retains metadata used by rendering.
struct FastEvent {
    EventType type=EventType::Tap;
    std::uint32_t tick=0,end_tick=0;
    std::uint8_t lane=0,target_lane=0,touch_sensor=0,strength=0,path_id=0;
    SourceLocation location{};
};
struct FastChart : maimoe::ParsedChart { std::vector<FastEvent> events; };

namespace {

struct LineMap { std::size_t start; std::uint32_t line, offset; };
struct Body {
    std::string bytes;
    std::vector<LineMap> lines;
    mutable std::size_t cursor=0;
    std::size_t size() const { return bytes.size(); }
    bool empty() const { return bytes.empty(); }
};

struct SourceKey {
    bool touch = false;
    std::uint8_t index = 0;

    auto operator<=>(const SourceKey&) const = default;
};

struct SourceState {
    bool have_tick = false;
    std::uint32_t tick = 0;
    std::uint32_t active_end = 0;
};

void add_candidate(FastChart& chart, SemanticError candidate) {
    const auto existing = std::find_if(
        chart.semantic_candidates.begin(), chart.semantic_candidates.end(),
        [&candidate](const SemanticError& error) { return error.code == candidate.code; });
    if (existing == chart.semantic_candidates.end()) {
        chart.semantic_candidates.push_back(candidate);
        return;
    }
    if (std::tuple(candidate.location.byte_offset, candidate.detail) <
        std::tuple(existing->location.byte_offset, existing->detail)) {
        *existing = candidate;
    }
}

struct BodyParser {
    FastChart& chart;
    const Body& body;
    std::uint64_t tick = 0;
    std::uint32_t division = 4;
    std::uint32_t bpm_milli = 0;
    std::uint32_t parsed_events = 0;
    bool initial_bpm_control = true;
    bool initial_division_control = true;
    bool conventional_division_pending = false;
    bool ended = false;
    std::array<SourceState, 8> button_sources{};
    std::array<SourceState, kTouchSensorCount> touch_sources{};

    void add_error(SemanticErrorCode code, SourceLocation location, std::uint32_t detail) {
        add_candidate(chart, SemanticError{code, location, detail});
    }

    void add_warning(WarningCode code, SourceLocation location) {
        chart.warnings.push_back(Warning{code, 1, location});
    }
};

[[nodiscard]] SourceLocation location_at(const Body& body,
                                         std::size_t index) noexcept {
    if (body.empty()) {
        return SourceLocation{};
    }
    index = std::min(index,body.size()-1U);
    while(body.cursor+1<body.lines.size() && index>=body.lines[body.cursor+1].start) ++body.cursor;
    while(index<body.lines[body.cursor].start) --body.cursor;
    const auto& line=body.lines[body.cursor];
    return SourceLocation{line.line,static_cast<std::uint32_t>(index-line.start+1U),
                          static_cast<std::uint32_t>(line.offset+index-line.start)};
}

[[nodiscard]] std::uint32_t saturated_u32(std::uint64_t value) noexcept {
    return value > std::numeric_limits<std::uint32_t>::max()
               ? std::numeric_limits<std::uint32_t>::max()
               : static_cast<std::uint32_t>(value);
}

template <typename UInt>
[[nodiscard]] UInt parse_uint(std::string_view text, SourceLocation location,
                              std::string_view what) {
    if (text.empty() || (text.size() > 1U && text.front() == '0')) {
        throw FormatError(std::string(what) + " is not a canonical unsigned decimal",
                          location.line, location.column);
    }
    UInt value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw FormatError(std::string(what) + " is malformed or out of range",
                          location.line, location.column);
    }
    return value;
}

[[nodiscard]] std::uint32_t parse_bpm_milli(std::string_view text, SourceLocation location) {
    const std::size_t dot = text.find('.');
    if (dot != std::string_view::npos && text.find('.', dot + 1U) != std::string_view::npos) {
        throw FormatError("BPM contains more than one decimal point", location.line,
                          location.column);
    }
    const std::string_view whole_text = text.substr(0, dot);
    const std::string_view fraction = dot == std::string_view::npos
                                          ? std::string_view{}
                                          : text.substr(dot + 1U);
    if (whole_text.empty() || (whole_text.size() > 1U && whole_text.front() == '0') ||
        (dot != std::string_view::npos && (fraction.empty() || fraction.size() > 3U))) {
        throw FormatError("BPM is not a canonical decimal", location.line, location.column);
    }
    const std::uint32_t whole = parse_uint<std::uint32_t>(whole_text, location, "BPM");
    std::uint32_t fractional = 0;
    if (!fraction.empty()) {
        if (fraction.back() == '0') {
            throw FormatError("BPM fractional digits have a trailing zero", location.line,
                              location.column);
        }
        for (const char digit : fraction) {
            if (digit < '0' || digit > '9') {
                throw FormatError("BPM contains a non-decimal character", location.line,
                                  location.column);
            }
            fractional = fractional * 10U + static_cast<std::uint32_t>(digit - '0');
        }
        if (fraction.size() == 1U) {
            fractional *= 100U;
        } else if (fraction.size() == 2U) {
            fractional *= 10U;
        }
    }
    if (whole > (std::numeric_limits<std::uint32_t>::max() - fractional) / 1000U) {
        throw FormatError("BPM is out of the milli-BPM lexical range", location.line,
                          location.column);
    }
    return whole * 1000U + fractional;
}

[[nodiscard]] std::string bpm_text(std::uint32_t milli) {
    std::string result = std::to_string(milli / 1000U);
    const std::uint32_t fraction = milli % 1000U;
    if (fraction != 0U) {
        std::string digits = std::to_string(1000U + fraction).substr(1U);
        while (digits.back() == '0') {
            digits.pop_back();
        }
        result.push_back('.');
        result += digits;
    }
    return result;
}

[[nodiscard]] std::uint8_t touch_sensor(std::string_view text, SourceLocation location) {
    if (text == "C") {
        return 16U;
    }
    if (text.size() != 2U || text[1] < '1' || text[1] > '8') {
        throw FormatError("touch sensor must be A1..A8, B1..B8, C, D1..D8, or E1..E8",
                          location.line, location.column);
    }
    const std::uint8_t ring_index = static_cast<std::uint8_t>(text[1] - '1');
    switch (text[0]) {
    case 'A':
        return ring_index;
    case 'B':
        return static_cast<std::uint8_t>(8U + ring_index);
    case 'D':
        return static_cast<std::uint8_t>(17U + ring_index);
    case 'E':
        return static_cast<std::uint8_t>(25U + ring_index);
    default:
        throw FormatError("unsupported touch sensor ring", location.line, location.column);
    }
}

[[nodiscard]] std::string touch_name(std::uint8_t sensor) {
    if (sensor < 8U) {
        return std::string{"A"} + static_cast<char>('1' + sensor);
    }
    if (sensor < 16U) {
        return std::string{"B"} + static_cast<char>('1' + sensor - 8U);
    }
    if (sensor == 16U) {
        return "C";
    }
    if (sensor < 25U) {
        return std::string{"D"} + static_cast<char>('1' + sensor - 17U);
    }
    if (sensor < kTouchSensorCount) {
        return std::string{"E"} + static_cast<char>('1' + sensor - 25U);
    }
    throw FormatError("touch sensor index is outside 0..32");
}

[[nodiscard]] SourceKey event_source(const FastEvent& event) noexcept {
    if (event.type == EventType::TouchTap || event.type == EventType::TouchHold) {
        return SourceKey{true, event.touch_sensor};
    }
    return SourceKey{false, event.lane};
}

[[nodiscard]] std::uint32_t source_detail(SourceKey source) noexcept {
    return source.touch ? 0x100U + source.index : source.index;
}

struct CanonicalPrefix {
    std::uint32_t tick=0;
    std::uint64_t hash=0;
    std::size_t length=0;
};
[[nodiscard]] std::uint64_t canonical_word(const FastEvent& event, CanonicalPrefix& prefix) {
    char buffer[96];
    char* p = buffer;
    auto number = [&](std::uint32_t n) { p = std::to_chars(p, p + 10, n).ptr; };
    auto literal = [&](std::string_view s) { std::memcpy(p,s.data(),s.size()); p += s.size(); };
    if(prefix.length==0 || prefix.tick!=event.tick) {
        number(event.tick); *p++=' ';
        prefix.tick=event.tick;prefix.length=p-buffer;prefix.hash=0x243f6a8885a308d3ULL;
        for(const char* q=buffer;q!=p;++q) prefix.hash=std::rotl(prefix.hash^static_cast<unsigned char>(*q),5)*0x9e3779b185ebca87ULL+0x165667b19e3779f9ULL;
        p=buffer;
    }
    auto sensor=[&]{
        const unsigned n=event.touch_sensor;
        if(n==16){*p++='C';return;}
        if(n<8){*p++='A';*p++=char('1'+n);}
        else if(n<16){*p++='B';*p++=char('1'+n-8);}
        else if(n<25){*p++='D';*p++=char('1'+n-17);}
        else {*p++='E';*p++=char('1'+n-25);}
    };
    switch (event.type) {
    case EventType::Tap: literal("TAP B"); *p++=static_cast<char>('1'+event.lane); literal(" 128"); break;
    case EventType::Break: literal("BREAK B"); *p++=static_cast<char>('1'+event.lane); literal(" 255 3"); break;
    case EventType::Hold: literal("HOLD B"); *p++=static_cast<char>('1'+event.lane); *p++=' '; number(event.end_tick); literal(" 128"); break;
    case EventType::Slide: literal("SLIDE B"); *p++=static_cast<char>('1'+event.lane); *p++=' '; number(event.end_tick); literal(" B"); *p++=static_cast<char>('1'+event.target_lane); literal(" 0"); break;
    case EventType::TouchTap: literal("TOUCH "); sensor(); literal(" 192"); break;
    case EventType::TouchHold: literal("TOUCH_HOLD "); sensor(); *p++=' '; number(event.end_tick); literal(" 192"); break;
    }
    std::uint64_t hash=prefix.hash;
    for (const char* q=buffer;q!=p;++q) hash=std::rotl(hash^static_cast<unsigned char>(*q),5)*0x9e3779b185ebca87ULL+0x165667b19e3779f9ULL;
    return hash ^ static_cast<std::uint64_t>(p-buffer+prefix.length);
}

[[nodiscard]] std::uint64_t parse_duration(BodyParser& parser, std::string_view token,
                                           std::size_t token_offset) {
    const SourceLocation location = location_at(parser.body, token_offset);
    const std::size_t colon = token.find(':');
    if (colon == std::string_view::npos || token.find(':', colon + 1U) != std::string_view::npos) {
        throw FormatError("duration must be one canonical D:N pair", location.line,
                          location.column);
    }
    const std::uint32_t denominator = parse_uint<std::uint32_t>(
        token.substr(0, colon), location, "duration denominator");
    const std::uint32_t numerator = parse_uint<std::uint32_t>(
        token.substr(colon + 1U), location, "duration numerator");
    if (denominator == 0U || numerator == 0U) {
        throw FormatError("duration components must be nonzero", location.line, location.column);
    }
    if (std::gcd(denominator, numerator) != 1U) {
        throw FormatError("duration D:N must be reduced", location.line, location.column);
    }
    const std::uint64_t scaled = static_cast<std::uint64_t>(kTicksPerWhole) * numerator;
    if (scaled % denominator != 0U) {
        throw FormatError("duration is not exact on the 384-tick grid", location.line,
                          location.column);
    }
    return scaled / denominator;
}

void increment_count(FastChart& chart, EventType type) {
    switch (type) {
    case EventType::Tap:
        ++chart.counts.tap;
        break;
    case EventType::Break:
        ++chart.counts.break_count;
        break;
    case EventType::Hold:
        ++chart.counts.hold;
        break;
    case EventType::Slide:
        ++chart.counts.slide;
        break;
    case EventType::TouchTap:
    case EventType::TouchHold:
        ++chart.counts.touch;
        break;
    }
    ++chart.counts.total;
}

void parse_note(BodyParser& parser, std::string_view note, std::size_t note_offset) {
    const SourceLocation source = location_at(parser.body, note_offset);
    if (note.empty()) {
        throw FormatError("empty note in explicit chord", source.line, source.column);
    }

    FastEvent event;
    event.tick = saturated_u32(parser.tick);
    event.location = source;
    const bool button = note.front() >= '1' && note.front() <= '8';
    std::size_t sensor_length = 0;
    if (button) {
        event.lane = static_cast<std::uint8_t>(note.front() - '1');
        sensor_length = 1;
    } else if (note.front() == 'C') {
        event.touch_sensor = touch_sensor(note.substr(0, 1), source);
        sensor_length = 1;
    } else if (note.front() == 'A' || note.front() == 'B' || note.front() == 'D' ||
               note.front() == 'E') {
        if (note.size() < 2U) {
            throw FormatError("truncated touch sensor", source.line, source.column);
        }
        event.touch_sensor = touch_sensor(note.substr(0, 2), source);
        sensor_length = 2;
    } else {
        throw FormatError("unsupported note syntax", source.line, source.column);
    }

    if (note.size() == sensor_length) {
        event.type = button ? EventType::Tap : EventType::TouchTap;
        event.strength = button ? 128U : 192U;
    } else if (button && note.size() == 2U && note[1] == 'b') {
        event.type = EventType::Break;
        event.strength = 255U;
    } else {
        const std::size_t open = note.find('[', sensor_length);
        if (open == std::string_view::npos || note.back() != ']' ||
            note.find('[', open + 1U) != std::string_view::npos ||
            note.find(']', open + 1U) != note.size() - 1U) {
            throw FormatError("timed note requires exactly one [D:N] suffix", source.line,
                              source.column);
        }
        const std::string_view head = note.substr(sensor_length, open - sensor_length);
        const std::uint64_t duration = parse_duration(
            parser, note.substr(open + 1U, note.size() - open - 2U), note_offset + open + 1U);
        const std::uint64_t end_tick = parser.tick + duration;
        event.end_tick = saturated_u32(end_tick);
        parser.chart.max_tick = std::max(parser.chart.max_tick, event.end_tick);
        if (end_tick > kMaxTimelineTick) {
            parser.add_error(SemanticErrorCode::TimelineRange, source,
                             saturated_u32(end_tick));
        }
        if (head == "h") {
            event.type = button ? EventType::Hold : EventType::TouchHold;
            event.strength = button ? 128U : 192U;
        } else if (button && head.size() == 2U && head.front() == '-' &&
                   head.back() >= '1' && head.back() <= '8') {
            event.type = EventType::Slide;
            event.target_lane = static_cast<std::uint8_t>(head.back() - '1');
            if (event.target_lane == event.lane) {
                parser.add_error(SemanticErrorCode::SlideSelf, source,
                                 static_cast<std::uint32_t>(event.lane + 1U));
            }
        } else {
            throw FormatError(button ? "unsupported button suffix"
                                     : "touch notes support only tap and h[D:N]",
                              source.line, source.column + sensor_length);
        }
    }

    if (parser.tick > kMaxTimelineTick) {
        parser.add_error(SemanticErrorCode::TimelineRange, source,
                         saturated_u32(parser.tick));
    }
    const SourceKey key = event_source(event);
    SourceState& source_state = key.touch ? parser.touch_sources[key.index]
                                          : parser.button_sources[key.index];
    if (source_state.have_tick && source_state.tick == event.tick) {
        parser.add_error(SemanticErrorCode::DuplicateSource, source, source_detail(key));
    } else if (source_state.active_end > event.tick) {
        parser.add_error(SemanticErrorCode::ActiveOverlap, source, source_state.active_end);
    }
    source_state.have_tick = true;
    source_state.tick = event.tick;
    if (event.type == EventType::Hold || event.type == EventType::Slide ||
        event.type == EventType::TouchHold) {
        source_state.active_end = std::max(source_state.active_end, event.end_tick);
    }
    increment_count(parser.chart, event.type);
    ++parser.parsed_events;
    if (parser.parsed_events == kMaxEvents + 1U) {
        parser.add_error(SemanticErrorCode::EventLimit, source, parser.parsed_events);
    }
    if (parser.parsed_events <= kMaxEvents) {
        // Canonical bytes are formatted on the stack when the event is consumed.
        parser.chart.events.push_back(std::move(event));
    }
}

void parse_controls(BodyParser& parser, std::string_view cell, std::size_t cell_offset,
                    std::size_t& cursor) {
    while (cursor < cell.size() && (cell[cursor] == '(' || cell[cursor] == '{')) {
        const char opening = cell[cursor];
        const char closing = opening == '(' ? ')' : '}';
        const std::size_t close = cell.find(closing, cursor + 1U);
        const SourceLocation location = location_at(parser.body, cell_offset + cursor);
        if (close == std::string_view::npos) {
            throw FormatError("unterminated timing control", location.line, location.column);
        }
        const std::string_view value = cell.substr(cursor + 1U, close - cursor - 1U);
        if (opening == '(') {
            const std::uint32_t bpm = parse_bpm_milli(value, location);
            const bool conventional = parser.initial_bpm_control &&
                                      parser.initial_division_control && parser.tick == 0U &&
                                      bpm == parser.chart.metadata.whole_bpm_milli &&
                                      cell.substr(close + 1U, 3U) == "{4}";
            if (bpm == parser.bpm_milli && !conventional) {
                parser.add_warning(WarningCode::RedundantBpm, location);
            }
            if (bpm < kMinBpmMilli || bpm > kMaxBpmMilli) {
                parser.add_error(SemanticErrorCode::BpmRange, location, bpm);
            }
            parser.bpm_milli = bpm;
            parser.initial_bpm_control = false;
            parser.conventional_division_pending = conventional;
        } else {
            const std::uint32_t division = parse_uint<std::uint32_t>(value, location, "division");
            if (division == 0U || kTicksPerWhole % division != 0U) {
                throw FormatError("division must be nonzero and exact on the 384-tick grid",
                                  location.line, location.column);
            }
            const bool conventional = parser.conventional_division_pending && division == 4U;
            if (division == parser.division && !conventional) {
                parser.add_warning(WarningCode::RedundantDivision, location);
            }
            parser.division = division;
            parser.initial_division_control = false;
            parser.conventional_division_pending = false;
        }
        cursor = close + 1U;
    }
}

void parse_cell(BodyParser& parser, std::size_t begin, std::size_t end, bool has_comma) {
    const std::string_view cell(parser.body.bytes.data()+begin,end-begin);
    if (cell == "E") {
        if (has_comma || parser.ended) {
            const SourceLocation location = location_at(parser.body, begin);
            throw FormatError("E must be the final cell without a trailing comma", location.line,
                              location.column);
        }
        
        parser.ended = true;
        return;
    }
    if (parser.ended) {
        const SourceLocation location = location_at(parser.body, begin);
        throw FormatError("body content follows E", location.line, location.column);
    }

    
    std::size_t cursor = 0;
    parse_controls(parser, cell, begin, cursor);
    std::size_t note_count = 0;
    if (cursor < cell.size()) {
        std::size_t note_begin = cursor;
        while (note_begin <= cell.size()) {
            const std::size_t slash = cell.find('/', note_begin);
            const std::size_t note_end = slash == std::string_view::npos ? cell.size() : slash;
            parse_note(parser, std::string_view(cell).substr(note_begin, note_end - note_begin),
                       begin + note_begin);
            ++note_count;
            if (slash == std::string_view::npos) {
                break;
            }
            note_begin = slash + 1U;
        }
    }
    if (note_count >= kDenseEachThreshold) {
        parser.add_warning(WarningCode::DenseEach, location_at(parser.body, begin + cursor));
    }

    const std::uint64_t step = kTicksPerWhole / parser.division;
    parser.tick += step;
    if (parser.tick > kMaxTimelineTick) {
        const std::size_t position = has_comma && end < parser.body.size() ? end : begin;
        parser.add_error(SemanticErrorCode::TimelineRange, location_at(parser.body, position),
                         saturated_u32(parser.tick));
    }
    parser.chart.max_tick = std::max(parser.chart.max_tick, saturated_u32(parser.tick));
}

[[nodiscard]] bool valid_metadata_value(std::string_view value) noexcept {
    return !value.empty() && value.front() != ' ' && value.back() != ' ' &&
           std::all_of(value.begin(), value.end(), [](char character) {
               const auto byte = static_cast<unsigned char>(character);
               return byte >= 0x20U && byte <= 0x7eU;
           });
}

[[nodiscard]] std::vector<std::string_view> slow_strict_lines(std::string_view text) {
    if (text.empty() || text.size() > kMaxChartBytes || text.back() != '\n') {
        throw FormatError("maidata must be 1..8 MiB and end with LF");
    }
    std::vector<std::string_view> lines;
    std::size_t start = 0;
    std::size_t line_number = 1;
    for (std::size_t index = 0; index < text.size(); ++index) {
        const unsigned char byte = static_cast<unsigned char>(text[index]);
        if (byte != '\n' && (byte < 0x20U || byte > 0x7eU)) {
            throw FormatError("maidata must use printable ASCII and LF", line_number,
                              index - start + 1U);
        }
        if (byte != '\n') {
            continue;
        }
        const std::string_view line = text.substr(start, index - start);
        const std::size_t limit = line_number <= 7U ? kMaxMetadataLineBytes
                                                    : kMaxChartLineBytes;
        if (line.size() > limit) {
            throw FormatError("maidata line exceeds its byte limit", line_number, limit + 1U);
        }
        if (line.empty()) {
            throw FormatError("maidata does not permit empty physical lines", line_number, 1);
        }
        if (line.back() == ' ') {
            throw FormatError("maidata line has trailing whitespace", line_number, line.size());
        }
        lines.push_back(line);
        start = index + 1U;
        ++line_number;
    }
    return lines;
}

[[nodiscard]] std::vector<std::string_view> strict_lines(std::string_view text) {
    if(text.empty() || text.size()>kMaxChartBytes || text.back()!='\n') return slow_strict_lines(text);
    const auto* p=reinterpret_cast<const std::uint8_t*>(text.data());
    std::size_t i=0;
    for(;i+16<=text.size();i+=16) {
        const auto v=vld1q_u8(p+i);
        auto valid=vorrq_u8(vceqq_u8(v,vdupq_n_u8(10)),vandq_u8(vcgeq_u8(v,vdupq_n_u8(32)),vcleq_u8(v,vdupq_n_u8(126))));
        if(vminvq_u8(valid)!=255) return slow_strict_lines(text);
    }
    for(;i<text.size();++i) if(p[i]!=10 && (p[i]<32 || p[i]>126)) return slow_strict_lines(text);
    std::vector<std::string_view> lines;
    lines.reserve(text.size()/256+8);
    std::size_t start=0;
    while(start<text.size()) {
        const auto end=text.find('\n',start);
        const auto line=text.substr(start,end-start);
        const auto limit=lines.size()<7?kMaxMetadataLineBytes:kMaxChartLineBytes;
        if(line.empty() || line.size()>limit || line.back()==' ') return slow_strict_lines(text);
        lines.push_back(line);start=end+1;
    }
    return lines;
}

[[nodiscard]] std::string_view metadata_value(std::string_view line,
                                              std::string_view prefix,
                                              std::size_t line_number) {
    if (!line.starts_with(prefix)) {
        throw FormatError("metadata field is missing or out of order", line_number, 1);
    }
    const std::string_view value = line.substr(prefix.size());
    if (!valid_metadata_value(value)) {
        throw FormatError("metadata value must be nonempty canonical printable ASCII",
                          line_number, prefix.size() + 1U);
    }
    return value;
}

}

const FastChart& parse_maidata(std::string_view text) {
    const std::vector<std::string_view> lines = strict_lines(text);
    if (lines.size() < 8U) {
        throw FormatError("maidata requires seven metadata lines and a body", lines.size() + 1U,
                          1);
    }

    thread_local FastChart chart;
    chart.events.clear();chart.warnings.clear();chart.semantic_candidates.clear();
    chart.counts={};chart.max_tick=0;
    static_cast<void>(metadata_value(lines[0], "&title=", 1));
    static_cast<void>(metadata_value(lines[1], "&artist=", 2));
    static_cast<void>(metadata_value(lines[2], "&des=", 3));
    if (lines[3] != "&first=0") {
        throw FormatError("fourth metadata line must be exactly &first=0", 4, 1);
    }
    constexpr std::string_view whole_prefix = "&wholebpm=";
    if (!lines[4].starts_with(whole_prefix)) {
        throw FormatError("fifth metadata line must be &wholebpm=", 5, 1);
    }
    chart.metadata.whole_bpm_milli = parse_bpm_milli(
        lines[4].substr(whole_prefix.size()), SourceLocation{5, 11,
        static_cast<std::uint32_t>(text.find(whole_prefix) + whole_prefix.size())});
    if (chart.metadata.whole_bpm_milli < kMinBpmMilli ||
        chart.metadata.whole_bpm_milli > kMaxBpmMilli) {
        throw FormatError("wholebpm must be in 30..400 BPM", 5, 11);
    }
    static_cast<void>(metadata_value(lines[5], "&lv_1=", 6));
    if (lines[6] != "&inote_1=") {
        throw FormatError("seventh metadata line must be exactly &inote_1=", 7, 1);
    }

    thread_local Body body;
    body.bytes.clear();body.lines.clear();body.cursor=0;
    body.bytes.reserve(text.size());
    body.lines.reserve(lines.size()-7U);
    chart.events.reserve(text.size()/5U);
    std::size_t absolute_offset = 0;
    for (std::size_t line = 0; line < 7U; ++line) {
        absolute_offset += lines[line].size() + 1U;
    }
    for (std::size_t line = 7U; line < lines.size(); ++line) {
        if (line + 1U < lines.size() && lines[line].back() != ',') {
            throw FormatError("continued body lines must end at a comma", line + 1U,
                              lines[line].size());
        }
        const auto space = lines[line].find(' ');
        if (space != std::string_view::npos) throw FormatError("body does not permit whitespace",line+1U,space+1U);
        body.lines.push_back(LineMap{body.size(),static_cast<std::uint32_t>(line+1U),static_cast<std::uint32_t>(absolute_offset)});
        body.bytes.append(lines[line]);
        absolute_offset += lines[line].size() + 1U;
    }

    BodyParser parser{chart, body};
    parser.bpm_milli = chart.metadata.whole_bpm_milli;
    std::size_t cell_begin = 0;
    for (std::size_t index = 0; index <= body.size(); ++index) {
        if (index != body.size() && body.bytes[index] != ',') {
            continue;
        }
        parse_cell(parser, cell_begin, index, index != body.size());
        cell_begin = index + 1U;
    }
    if (!parser.ended) {
        const SourceLocation location = location_at(body, body.size() - 1U);
        throw FormatError("chart must terminate with E", location.line, location.column);
    }
    if (chart.events.empty() && chart.counts.total == 0U) {
        chart.warnings.push_back(Warning{WarningCode::EmptyChart, 1,
                                         location_at(body,0)});
    }

    // Ticks are monotone by construction. Only equal-tick events need sorting.
    if(chart.semantic_candidates.empty()) {
        auto& events=chart.events;
        for(std::size_t begin=0;begin<events.size();) {
            std::size_t end=begin+1;
            while(end<events.size() && events[end].tick==events[begin].tick) ++end;
            for(std::size_t i=begin+1;i<end;++i) {
                std::size_t j=i;
                while(j>begin && static_cast<unsigned>(events[j].type)<static_cast<unsigned>(events[j-1].type)) {
                    std::swap(events[j],events[j-1]); --j;
                }
            }
            begin=end;
        }
    }
    std::sort(chart.semantic_candidates.begin(), chart.semantic_candidates.end(),
              [](const SemanticError& lhs, const SemanticError& rhs) {
                  return std::tuple(lhs.location.byte_offset,
                                    static_cast<std::uint16_t>(lhs.code)) <
                         std::tuple(rhs.location.byte_offset,
                                    static_cast<std::uint16_t>(rhs.code));
              });
    chart.counts.warning = static_cast<std::uint32_t>(chart.warnings.size());
    return chart;
}


} // namespace maimoe::fast


#include <arm_neon.h>
namespace maimoe::fast {
constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};


// Kunpeng 920 implements ARMv8 SHA-256 instructions.
__attribute__((target("+crypto")))
Digest fast_sha256(std::span<const std::uint8_t> bytes) {
    const std::uint32_t initial[8]={0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
                                  0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};
    uint32x4_t a=vld1q_u32(initial), b=vld1q_u32(initial+4);
    std::uint8_t tail[128]{};
    const std::size_t full=bytes.size()/64, remain=bytes.size()%64;
    if(remain) std::memcpy(tail,bytes.data()+64*full,remain);
    tail[remain]=0x80;
    const std::size_t tailbytes=remain<56?64:128;
    const std::uint64_t bits=bytes.size()*8;
    for(unsigned j=0;j<8;j++) tail[tailbytes-1-j]=std::uint8_t(bits>>(8*j));
    for(std::size_t block=0;block<full+tailbytes/64;block++) {
        const std::uint8_t* p=block<full?bytes.data()+64*block:tail+64*(block-full);
        uint32x4_t m0=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p)));
        uint32x4_t m1=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p+16)));
        uint32x4_t m2=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p+32)));
        uint32x4_t m3=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p+48)));
        const auto saved_a=a,saved_b=b;
        #pragma GCC unroll 16
        for(unsigned round=0;round<16;round++) {
            auto k=vaddq_u32(m0,vld1q_u32(kRoundConstants.data()+round*4));
            auto old_a=a;
            a=vsha256hq_u32(a,b,k);
            b=vsha256h2q_u32(b,old_a,k);
            auto next=m0;
            if(round<12) next=vsha256su1q_u32(vsha256su0q_u32(m0,m1),m2,m3);
            m0=m1;m1=m2;m2=m3;m3=next;
        }
        a=vaddq_u32(a,saved_a);b=vaddq_u32(b,saved_b);
    }
    Digest result;
    vst1q_u8(result.data(),vrev32q_u8(vreinterpretq_u8_u32(a)));
    vst1q_u8(result.data()+16,vrev32q_u8(vreinterpretq_u8_u32(b)));
    return result;
}
}

namespace maimoe::fast {
namespace {
constexpr std::uint64_t kMixMultiplier = 0xd6e8feb86659fd93ULL;
constexpr std::uint64_t kMixIncrement = 0xa5a3564e27f8862fULL;

constexpr std::array<std::array<std::int16_t, 2>, kTouchSensorCount> kTouchPositions = {{
    {128, 8}, {173, 21}, {192, 72}, {173, 123}, {128, 136}, {83, 123}, {64, 72}, {83, 21},
    {128, 24}, {162, 38}, {176, 72}, {162, 106}, {128, 120}, {94, 106}, {80, 72}, {94, 38},
    {128, 72},
    {128, 41}, {150, 50}, {159, 72}, {150, 94}, {128, 103}, {106, 94}, {97, 72}, {106, 50},
    {128, 55}, {140, 60}, {145, 72}, {140, 84}, {128, 89}, {116, 84}, {111, 72}, {116, 60},
}};

void add_button_energy(State& state, std::uint8_t lane, std::uint32_t value) noexcept {
    if (lane >= state.energy.size()) {
        return;
    }
    state.energy[lane] = static_cast<std::uint16_t>(
        std::min<std::uint32_t>(65535U, static_cast<std::uint32_t>(state.energy[lane]) + value));
}

void add_touch_energy(State& state, std::uint8_t sensor, std::uint32_t value) noexcept {
    if (sensor >= state.touch_energy.size()) {
        return;
    }
    state.touch_energy[sensor] = static_cast<std::uint16_t>(
        std::min<std::uint32_t>(65535U,
            static_cast<std::uint32_t>(state.touch_energy[sensor]) + value));
}

void mix_event(State& state, const FastEvent& event, CanonicalPrefix& prefix) noexcept {
    state.phase = std::rotl(state.phase ^ canonical_word(event,prefix), 13) * kMixMultiplier +
                  kMixIncrement;
}

void apply_event(State& state, const FastEvent& event, CanonicalPrefix& prefix) noexcept {
    switch (event.type) {
    case EventType::Tap:
        if (event.lane < state.energy.size()) {
            add_button_energy(state, event.lane, event.strength);
            state.taps[event.lane] = LatestTap{true, event.strength};
        }
        break;
    case EventType::Hold:
        if (event.lane < state.energy.size()) {
            add_button_energy(state, event.lane,
                              std::max<std::uint32_t>(1U, event.strength / 4U));
            state.holds[event.lane] = ActiveHold{true, event.end_tick, event.strength};
        }
        break;
    case EventType::Slide:
        if (event.lane < state.energy.size() && event.target_lane < state.energy.size()) {
            const std::uint32_t distance = event.target_lane > event.lane
                                               ? event.target_lane - event.lane
                                               : event.lane - event.target_lane;
            add_button_energy(state, event.lane, 8U * (distance + 1U));
            state.slides[event.lane] = ActiveSlide{
                true, event.end_tick, event.target_lane, event.path_id};
        }
        break;
    case EventType::Break:
        if (event.lane < state.energy.size()) {
            add_button_energy(state, event.lane, 2U * event.strength);
            state.breaks[event.lane] = LatestBreak{true, event.strength, 3};
            state.break_remaining = 3;
        }
        break;
    case EventType::TouchTap:
        if (event.touch_sensor < state.touch_energy.size()) {
            add_touch_energy(state, event.touch_sensor, event.strength);
            state.touches[event.touch_sensor] = LatestTouch{true, false, event.strength};
        }
        break;
    case EventType::TouchHold:
        if (event.touch_sensor < state.touch_energy.size()) {
            add_touch_energy(state, event.touch_sensor,
                             std::max<std::uint32_t>(1U, event.strength / 4U));
            state.touch_holds[event.touch_sensor] = ActiveTouchHold{
                true, event.end_tick, event.strength};
            state.touches[event.touch_sensor] = LatestTouch{true, true, event.strength};
        }
        break;
    }
    mix_event(state, event,prefix);
}

template <typename Consumer>
void evolve_states_impl(std::uint64_t chart_id, const FastChart& chart,
                        Consumer&& consume) {
    State current;
    CanonicalPrefix prefix;
    const std::uint64_t bpm_q16 =
        (static_cast<std::uint64_t>(chart.metadata.whole_bpm_milli) * 65536ULL + 500ULL) /
        1000ULL;
    current.phase = 0x6a09e667f3bcc909ULL ^ chart_id ^ (bpm_q16 << 17U) ^
                    static_cast<std::uint64_t>(kTouchSensorCount);
    current.warning_count = static_cast<std::uint32_t>(chart.warnings.size());
    for (const Warning& warning : chart.warnings) {
        current.latest_warning_severity = warning.severity;
    }

    std::size_t next_event = 0;
    while (next_event < chart.events.size() && chart.events[next_event].tick == 0U) {
        apply_event(current, chart.events[next_event],prefix);
        ++next_event;
    }
    consume(0U, current);

    for (std::size_t frame = 0; frame + 1U < kFrameCount; ++frame) {
        State next = current;
        const std::uint32_t target_tick = frame_tick(chart.max_tick, frame + 1U);
        for (std::size_t lane = 0; lane < next.energy.size(); ++lane) {
            next.energy[lane] = static_cast<std::uint16_t>(
                (7U * static_cast<std::uint32_t>(next.energy[lane])) / 8U);
            if (current.holds[lane].present) {
                add_button_energy(next, static_cast<std::uint8_t>(lane),
                                  std::max<std::uint32_t>(
                                      1U, current.holds[lane].strength / 8U));
            }
            if (current.slides[lane].present) {
                const std::uint8_t target = current.slides[lane].target_lane;
                const std::uint32_t distance = target > lane
                                                   ? target - static_cast<std::uint32_t>(lane)
                                                   : static_cast<std::uint32_t>(lane) - target;
                add_button_energy(next, static_cast<std::uint8_t>(lane),
                                  4U * (distance + 1U));
            }
        }
        for (std::size_t sensor = 0; sensor < next.touch_energy.size(); ++sensor) {
            next.touch_energy[sensor] = static_cast<std::uint16_t>(
                (7U * static_cast<std::uint32_t>(next.touch_energy[sensor])) / 8U);
            if (current.touch_holds[sensor].present) {
                add_touch_energy(next, static_cast<std::uint8_t>(sensor),
                                 std::max<std::uint32_t>(
                                     1U, current.touch_holds[sensor].strength / 8U));
            }
        }
        if (next.break_remaining > 0U) {
            --next.break_remaining;
        }
        for (LatestBreak& note_break : next.breaks) {
            if (note_break.flash_frames > 0U) {
                --note_break.flash_frames;
            }
        }
        for (std::size_t lane = 0; lane < next.holds.size(); ++lane) {
            if (next.holds[lane].present && next.holds[lane].end_tick <= target_tick) {
                next.holds[lane] = ActiveHold{};
            }
            if (next.slides[lane].present && next.slides[lane].end_tick <= target_tick) {
                next.slides[lane] = ActiveSlide{};
            }
        }
        for (ActiveTouchHold& hold : next.touch_holds) {
            if (hold.present && hold.end_tick <= target_tick) {
                hold = ActiveTouchHold{};
            }
        }
        next.phase = std::rotl(next.phase ^ static_cast<std::uint64_t>(frame + 1U), 7) *
                         kMixMultiplier +
                     kMixIncrement;
        while (next_event < chart.events.size() && chart.events[next_event].tick <= target_tick) {
            apply_event(next, chart.events[next_event],prefix);
            ++next_event;
        }
        current = std::move(next);
        consume(frame + 1U, current);
    }
}

}
EndpointStates fast_evolve_endpoint_states(std::uint64_t chart_id, const FastChart& chart) {
    EndpointStates endpoints;
    evolve_states_impl(chart_id, chart, [&](std::size_t frame, const State& state) {
        if (frame == kFirstSampleFrame) {
            endpoints.frame_begin = state;
        } else if (frame == kLastSampleFrame) {
            endpoints.frame_end = state;
        }
    });
    return endpoints;
}

}
namespace maimoe::fast {
namespace {
struct Color {
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
    std::uint8_t a = 255;
};

struct Surface {
    std::span<std::uint8_t> pixels;
    int width = 0;
    int height = 0;
    bool sampled = false;
};

[[nodiscard]] Color gray(std::uint32_t value) noexcept {
    const auto byte = static_cast<std::uint8_t>(std::min<std::uint32_t>(255U, value));
    return Color{byte, byte, byte, 255};
}

[[nodiscard]] std::size_t pixel_offset(const Surface& surface, int x, int y) noexcept {
    return (static_cast<std::size_t>(y) * static_cast<std::size_t>(surface.width) +
            static_cast<std::size_t>(x)) * 4U;
}

void blend_at(Surface& surface, int x, int y, Color source,
              std::uint8_t state_alpha) noexcept {
    if (x < 0 || x >= surface.width || y < 0 || y >= surface.height) {
        return;
    }
    const std::uint32_t alpha = source.a;
    const std::size_t offset = pixel_offset(surface, x, y);
    const std::array<std::uint8_t, 3> source_components{source.r, source.g, source.b};
    for (std::size_t component = 0; component < source_components.size(); ++component) {
        const std::uint32_t destination = surface.pixels[offset + component];
        const std::uint32_t value =
            (static_cast<std::uint32_t>(source_components[component]) * alpha +
             destination * (255U - alpha) + 127U) /
            255U;
        surface.pixels[offset + component] = static_cast<std::uint8_t>(value);
    }
    surface.pixels[offset + 3U] = 255U;
}

void blend(Surface& surface, int x, int y, Color source,
           std::uint8_t state_alpha) noexcept {
    if (x < 0 || x >= kCanvasWidth || y < 0 || y >= kCanvasHeight) {
        return;
    }
    if (!surface.sampled) {
        blend_at(surface, x, y, source, state_alpha);
        return;
    }
    if ((x & 3) != 2 || (y & 3) != 2) {
        return;
    }
    blend_at(surface, (x - 2) / 4, (y - 2) / 4, source, state_alpha);
}

void rectangle(Surface& surface, int x0, int y0, int x1, int y1, Color color,
                std::uint8_t alpha) noexcept {
    x0 = std::max(x0, 0);
    y0 = std::max(y0, 0);
    x1 = std::min(x1, static_cast<int>(kCanvasWidth) - 1);
    y1 = std::min(y1, static_cast<int>(kCanvasHeight) - 1);
    if (x0 > x1 || y0 > y1) {
        return;
    }
    if (!surface.sampled) {
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                blend_at(surface, x, y, color, alpha);
            }
        }
        return;
    }
    const int first_x = (x0 + 1) / 4;
    const int first_y = (y0 + 1) / 4;
    for (int y = first_y; y < surface.height && 4 * y + 2 <= y1; ++y) {
        for (int x = first_x; x < surface.width && 4 * x + 2 <= x1; ++x) {
            blend_at(surface, x, y, color, alpha);
        }
    }
}

// The state machine never changes mirror, rotation, speed or global alpha.
[[nodiscard]] int lane_x(const State&,std::size_t lane) noexcept {return 16+32*lane;}
[[nodiscard]] std::uint16_t visible_energy(const State& state,std::size_t lane) noexcept {return state.energy[lane];}
[[nodiscard]] std::uint16_t visible_touch_energy(const State& state,std::size_t sensor) noexcept {return state.touch_energy[sensor];}

void draw_slide_point(Surface& surface, int center_x, int center_y, Color color,
                       std::uint8_t alpha) noexcept {
    rectangle(surface, center_x - 1, center_y - 1, center_x + 1, center_y + 1,
              color, alpha);
}

void fast_render_output(std::uint64_t chart_id, const ParsedChart& chart, const State& state,
                 std::size_t frame, std::span<std::uint8_t,kOutputFramePayloadBytes> output) {
    Surface surface{output,kOutputFrameWidth,kOutputFrameHeight,true};
    const bool sampled=true;
    auto pixels=output;
    const std::uint32_t c0 = static_cast<std::uint32_t>(chart_id & 0xffU);
    const std::uint32_t c1 = static_cast<std::uint32_t>((chart_id >> 8U) & 0xffU);
    const std::uint32_t c2 = static_cast<std::uint32_t>((chart_id >> 16U) & 0xffU);

    const std::uint8_t indices[16]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    const auto idx=vld1q_u8(indices);
    uint8x16x4_t rgba;
    rgba.val[3]=vdupq_n_u8(255);
    if((chart_id&7U)!=0U) {
        const auto groups=vshrq_n_u8(idx,2);
        const auto offsets=vmulq_u8(groups,vdupq_n_u8(13));
        for(unsigned y=0;y<36;++y) {
            const unsigned base=29U*(y/2)+7U*frame+c0;
            for(unsigned block=0;block<4;++block) {
                const auto q=vaddq_u8(offsets,vdupq_n_u8(base+52U*block));
                rgba.val[0]=q;
                rgba.val[1]=vaddq_u8(vmulq_u8(q,vdupq_n_u8(3)),vdupq_n_u8(17+c1));
                rgba.val[2]=vaddq_u8(vmulq_u8(q,vdupq_n_u8(5)),vdupq_n_u8(29+c2));
                vst4q_u8(output.data()+y*256+block*64,rgba);
            }
        }
    } else {
        const auto x=vaddq_u8(vshlq_n_u8(idx,2),vdupq_n_u8(2));
        const auto rx=vmulq_u8(x,vdupq_n_u8(17));
        const auto gx=vmulq_u8(x,vdupq_n_u8(29));
        const auto bx=vmulq_u8(x,vdupq_n_u8(7));
        for(unsigned y=0;y<36;++y) {
            const unsigned cy=4*y+2;
            for(unsigned block=0;block<4;++block) {
                rgba.val[0]=vaddq_u8(rx,vdupq_n_u8(17*64*block+31*cy+7*frame+c0));
                rgba.val[1]=vaddq_u8(gx,vdupq_n_u8(29*64*block+11*cy+13*frame+c1));
                rgba.val[2]=vaddq_u8(bx,vdupq_n_u8(7*64*block+19*cy+3*frame+c2));
                vst4q_u8(output.data()+y*256+block*64,rgba);
            }
        }
    }

    for (std::size_t lane = 0; lane < 8U; ++lane) {
        rectangle(surface, lane_x(state, lane), 0, lane_x(state, lane), kCanvasHeight - 1,
                  gray(48), std::uint8_t{255});
    }
    for (std::size_t lane = 0; lane < 8U; ++lane) {
        const std::uint16_t energy = visible_energy(state, lane);
        const int height = static_cast<int>((static_cast<std::uint32_t>(energy) * kCanvasHeight) / 65535U);
        if (height > 0) {
            const int x = lane_x(state, lane);
            rectangle(surface, x - 2, kCanvasHeight - height, x + 2, kCanvasHeight - 1,
                      gray(64U + (energy >> 10U)), std::uint8_t{255});
        }
    }
    const std::uint32_t tick = frame_tick(chart.max_tick, frame);
    for (std::size_t lane = 0; lane < 8U; ++lane) {
        const ActiveHold& hold = state.holds[lane];
        if (hold.present && hold.end_tick > tick) {
            const std::uint32_t remaining = hold.end_tick - tick;
            const std::uint32_t height = 1U +
                static_cast<std::uint32_t>((static_cast<std::uint64_t>(remaining) * 143U) /
                                           std::max<std::uint32_t>(1U, chart.max_tick));
            const int x = lane_x(state, lane);
            rectangle(surface, x - 1, 143 - static_cast<int>(std::min<std::uint32_t>(143U, height)),
                      x + 1, 143, gray(160U + (hold.strength >> 3U)), std::uint8_t{255});
        }
    }
    for (std::size_t lane = 0; lane < 8U; ++lane) {
        const ActiveSlide& slide = state.slides[lane];
        if (!slide.present || slide.target_lane >= 8U) {
            continue;
        }
        const int start_x = lane_x(state, lane);
        const int target_x = lane_x(state, slide.target_lane);
        for (int j = 0; j <= 31; ++j) {
            const int x = start_x + ((target_x - start_x) * j) / 31;
            const int y = 143 - (j * 143) / 31;
            draw_slide_point(surface, x, y, gray(176U + slide.path_id * 4U), std::uint8_t{255});
        }
    }
    for (std::size_t sensor = 0; sensor < kTouchSensorCount; ++sensor) {
        const auto position = touch_sensor_position(static_cast<std::uint8_t>(sensor));
        const int x = position[0];
        const int y = position[1];
        const std::uint16_t energy = visible_touch_energy(state, sensor);
        if (energy > 0U) {
            const int radius = 1 + static_cast<int>(energy >> 14U);
            rectangle(surface, x - radius, y - radius, x + radius, y + radius,
                      Color{64U, 208U, 224U,
                            static_cast<std::uint8_t>(128U + (energy >> 9U))},
                      std::uint8_t{255});
        }
        const ActiveTouchHold& hold = state.touch_holds[sensor];
        if (hold.present && hold.end_tick > tick) {
            const std::uint32_t remaining = hold.end_tick - tick;
            const int arm = 2 + static_cast<int>((static_cast<std::uint64_t>(remaining) * 5U) /
                                                  std::max<std::uint32_t>(1U, chart.max_tick));
            rectangle(surface, x - arm, y, x + arm, y,
                      Color{64U, 240U, 192U, 224U}, std::uint8_t{255});
            rectangle(surface, x, y - arm, x, y + arm,
                      Color{64U, 240U, 192U, 224U}, std::uint8_t{255});
        }
        if (state.touches[sensor].present) {
            const Color color = state.touches[sensor].hold
                                    ? Color{96U, 255U, 176U, 240U}
                                    : Color{96U, 224U, 255U, 240U};
            blend(surface, x, y, color, std::uint8_t{255});
            blend(surface, x - 1, y, color, std::uint8_t{255});
            blend(surface, x + 1, y, color, std::uint8_t{255});
            blend(surface, x, y - 1, color, std::uint8_t{255});
            blend(surface, x, y + 1, color, std::uint8_t{255});
        }
    }
    for (std::size_t lane = 0; lane < 8U; ++lane) {
        if (state.taps[lane].present) {
            const int x = lane_x(state, lane);
            rectangle(surface, x - 3, 71, x + 3, 73,
                      gray(208U + (state.taps[lane].strength >> 4U)), std::uint8_t{255});
        }
    }
    for (std::size_t lane = 0; lane < 8U; ++lane) {
        if (!state.breaks[lane].present) {
            continue;
        }
        const int x = lane_x(state, lane);
        const Color color = gray(240U + (state.breaks[lane].strength >> 6U));
        rectangle(surface, x - 4, 48, x + 4, 48, color, std::uint8_t{255});
        if (state.breaks[lane].flash_frames > 0U) {
            blend(surface, x, 44, color, std::uint8_t{255});
            blend(surface, x, 45, color, std::uint8_t{255});
            blend(surface, x, 46, color, std::uint8_t{255});
            blend(surface, x, 47, color, std::uint8_t{255});
            blend(surface, x, 49, color, std::uint8_t{255});
            blend(surface, x, 50, color, std::uint8_t{255});
            blend(surface, x, 51, color, std::uint8_t{255});
            blend(surface, x, 52, color, std::uint8_t{255});
        }
    }
    std::uint8_t break_flash_frames = 0;
    for (const LatestBreak& note_break : state.breaks) {
        break_flash_frames = std::max(break_flash_frames, note_break.flash_frames);
    }
    if (break_flash_frames > 0U) {
        const Color color = gray(224U + break_flash_frames);
        rectangle(surface, 0, 0, kCanvasWidth - 1, 0, color, std::uint8_t{255});
        rectangle(surface, 0, kCanvasHeight - 1, kCanvasWidth - 1, kCanvasHeight - 1,
                   color, std::uint8_t{255});
        rectangle(surface, 0, 1, 0, kCanvasHeight - 2, color, std::uint8_t{255});
        rectangle(surface, kCanvasWidth - 1, 1, kCanvasWidth - 1, kCanvasHeight - 2,
                  color, std::uint8_t{255});
    }
    if (state.warning_count > 0U) {
        const Color color = gray(192U + 16U * state.latest_warning_severity);
        for (std::uint32_t x = 0; x < std::min<std::uint32_t>(kCanvasWidth, state.warning_count); ++x) {
            blend(surface, static_cast<int>(x), 0, color, std::uint8_t{255});
        }
    }
    for (std::size_t j = 0; j < 8U; ++j) {
        const std::uint8_t byte = static_cast<std::uint8_t>((state.phase >> (8U * j)) & 0xffU);
        rectangle(surface, static_cast<int>(8U + j), 143 - (byte % 16U),
                  static_cast<int>(9U + j), 144 - (byte % 16U),
                  gray(32U + (byte % 224U)), std::uint8_t{255});
    }
}
}
}
namespace maimoe::fast {
using namespace maimoe::engine;
constexpr std::array<std::uint8_t, 8> kFrameMagic = {'M', 'M', 'F', 'R', 'M', 0, 0, 3};
constexpr std::array<std::uint8_t, 8> kExpectedMagic = {'M', 'A', 'I', 'E', 'X', 'P', 0, 3};
constexpr std::array<std::uint8_t, 4> kResultMagic = {'M', 'M', 'R', '3'};
constexpr std::uint16_t kExpectedHeaderBytes = 64U;
constexpr std::uint16_t kExpectedVersion = 3U;

void append_digest(Bytes& bytes, const Digest& digest) {
    bytes.insert(bytes.end(), digest.begin(), digest.end());
}

template <typename UInt>
void append_le_into(std::span<std::uint8_t> output, std::size_t& offset, UInt value) {
    static_assert(std::is_unsigned_v<UInt>);
    for (std::size_t i = 0; i < sizeof(UInt); ++i) {
        output[offset++] = static_cast<std::uint8_t>(value & static_cast<UInt>(0xffU));
        value >>= 8U;
    }
}

void append_digest_into(std::span<std::uint8_t> output, std::size_t& offset,
                        const Digest& digest) {
    std::memcpy(output.data() + offset, digest.data(), digest.size());
    offset += digest.size();
}

void fill_stem(std::array<char, 16>& stem, std::uint64_t chart_id) noexcept {
    static constexpr char kHexDigits[] = "0123456789abcdef";
    for (std::size_t index = 0; index < stem.size(); ++index) {
        stem[stem.size() - 1U - index] = kHexDigits[static_cast<std::size_t>(chart_id & 0xfU)];
        chart_id >>= 4U;
    }
}

[[nodiscard]] Digest read_digest(std::span<const std::uint8_t> bytes, std::size_t& offset) {
    if (offset > bytes.size() || bytes.size() - offset < Digest{}.size()) {
        throw FormatError("truncated digest");
    }
    Digest result{};
    std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(offset), result.size(), result.begin());
    offset += result.size();
    return result;
}

void require_zero(std::span<const std::uint8_t> bytes, std::size_t offset,
                  std::size_t count, std::string_view field) {
    if (offset > bytes.size() || bytes.size() - offset < count ||
        std::any_of(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                    bytes.begin() + static_cast<std::ptrdiff_t>(offset + count),
                    [](std::uint8_t value) { return value != 0U; })) {
        throw FormatError(std::string(field) + " must be zero");
    }
}

[[nodiscard]] Bytes read_file(const std::filesystem::path& path,
                              std::uint64_t expected_bytes) {
    if (expected_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw FormatError("file is too large for this platform");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw FormatError("cannot open " + path.string());
    }
    Bytes bytes(static_cast<std::size_t>(expected_bytes));
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (input.gcount() != static_cast<std::streamsize>(bytes.size()) ||
        input.peek() != std::char_traits<char>::eof()) {
        throw FormatError("file size changed while reading " + path.string());
    }
    return bytes;
}

[[nodiscard]] Bytes canonical_error_fields(const SemanticError& error) {
    if (!valid_semantic_error_code(error.code) || error.location.line == 0U ||
        error.location.column == 0U ||
        error.location.column > std::numeric_limits<std::uint16_t>::max()) {
        throw FormatError("semantic error cannot be encoded");
    }
    Bytes fields;
    fields.reserve(16U);
    append_le(fields, static_cast<std::uint16_t>(error.code));
    append_le(fields, static_cast<std::uint16_t>(error.location.column));
    append_le(fields, error.location.line);
    append_le(fields, error.detail);
    append_le(fields, std::uint32_t{0});
    return fields;
}

[[nodiscard]] Digest valid_result_digest(std::uint64_t chart_id, const Counts& counts,
                                         std::span<const std::uint8_t> state9,
                                         const Digest& frame_begin, const Digest& frame_end) {
    Bytes input;
    input.reserve(256U + state9.size());
    append_lp32(input, "MaiMoeResult/v3");
    append_le(input, chart_id);
    append_le(input, counts.total);
    append_le(input, counts.tap);
    append_le(input, counts.slide);
    append_le(input, counts.hold);
    append_le(input, counts.touch);
    append_le(input, counts.break_count);
    append_le(input, counts.warning);
    append_lp32(input, state9);
    append_lp32(input, std::span<const std::uint8_t>(frame_begin));
    append_lp32(input, std::span<const std::uint8_t>(frame_end));
    return fast_sha256(input);
}

[[nodiscard]] Bytes error_payload(std::uint64_t chart_id, FrameRole role,
                                  std::span<const std::uint8_t> fields) {
    Bytes seed_input;
    append_lp32(seed_input, "MaiMoeErrorFrame/v3");
    append_le(seed_input, chart_id);
    seed_input.push_back(static_cast<std::uint8_t>(role));
    append_lp32(seed_input, fields);
    const Digest seed = fast_sha256(seed_input);
    Bytes payload;
    payload.reserve(kOutputFramePayloadBytes);
    while (payload.size() < kOutputFramePayloadBytes) {
        payload.insert(payload.end(), seed.begin(), seed.end());
    }
    return payload;
}

[[nodiscard]] Digest error_result_digest(std::uint64_t chart_id,
                                         std::span<const std::uint8_t> fields,
                                         const Digest& frame_begin, const Digest& frame_end) {
    Bytes input;
    append_lp32(input, "MaiMoeError/v3");
    append_le(input, chart_id);
    append_lp32(input, fields);
    append_lp32(input, std::span<const std::uint8_t>(frame_begin));
    append_lp32(input, std::span<const std::uint8_t>(frame_end));
    return fast_sha256(input);
}

void encode_frame_into(std::span<std::uint8_t> output, std::uint64_t chart_id, FrameRole role,
                       ResultStatus status, std::span<const std::uint8_t> payload,
                       const Digest& digest) {
    if ((role != FrameRole::Begin && role != FrameRole::End) ||
        payload.size() != kOutputFramePayloadBytes || output.size() != kOutputFrameFileBytes) {
        throw FormatError("frame role or payload size is invalid");
    }
    std::size_t offset = 0;
    std::memcpy(output.data(), kFrameMagic.data(), kFrameMagic.size());
    offset += kFrameMagic.size();
    append_le_into(output, offset, static_cast<std::uint16_t>(kOutputFrameHeaderBytes));
    append_le_into(output, offset, kOutputFrameWidth);
    append_le_into(output, offset, kOutputFrameHeight);
    output[offset++] = 2U;
    output[offset++] = static_cast<std::uint8_t>(role);
    append_le_into(output, offset, chart_id);
    output[offset++] = static_cast<std::uint8_t>(status);
    output[offset++] = 0U;
    append_le_into(output, offset, std::uint16_t{0});
    append_le_into(output, offset, kOutputFramePayloadBytes);
    append_digest_into(output, offset, digest);
    std::memcpy(output.data() + offset, payload.data(), payload.size());
    offset += payload.size();
    if (offset != kOutputFrameFileBytes) {
        throw FormatError("internal frame encoder size mismatch");
    }
}

[[nodiscard]] Bytes encode_frame(std::uint64_t chart_id, FrameRole role,
                                 ResultStatus status, std::span<const std::uint8_t> payload,
                                 const Digest& digest) {
    Bytes output(kOutputFrameFileBytes);
    encode_frame_into(std::span<std::uint8_t>(output.data(), output.size()), chart_id, role,
                      status, payload, digest);
    return output;
}

void encode_result_into(std::span<std::uint8_t> output, std::uint64_t chart_id,
                        ResultStatus status, const Counts& counts,
                        std::span<const std::uint8_t> error_fields, const Digest& frame_begin,
                        const Digest& frame_end, const Digest& result) {
    if ((status == ResultStatus::Valid && !error_fields.empty()) ||
        (status == ResultStatus::Error && error_fields.size() != 16U) ||
        output.size() != kOutputResultFileBytes) {
        throw FormatError("result union fields are inconsistent");
    }
    std::size_t offset = 0;
    std::memcpy(output.data(), kResultMagic.data(), kResultMagic.size());
    offset += kResultMagic.size();
    output[offset++] = static_cast<std::uint8_t>(status);
    output[offset++] = 0U;
    append_le_into(output, offset, static_cast<std::uint16_t>(kOutputResultFileBytes));
    append_le_into(output, offset, chart_id);
    append_le_into(output, offset, counts.total);
    append_le_into(output, offset, counts.tap);
    append_le_into(output, offset, counts.slide);
    append_le_into(output, offset, counts.hold);
    append_le_into(output, offset, counts.touch);
    append_le_into(output, offset, counts.break_count);
    append_le_into(output, offset, counts.warning);
    if (error_fields.empty()) {
        std::fill_n(output.data() + offset, 16U, std::uint8_t{0});
        offset += 16U;
    } else {
        std::memcpy(output.data() + offset, error_fields.data(), error_fields.size());
        offset += error_fields.size();
    }
    append_digest_into(output, offset, frame_begin);
    append_digest_into(output, offset, frame_end);
    append_digest_into(output, offset, result);
    std::fill_n(output.data() + offset, 4U, std::uint8_t{0});
    offset += 4U;
    if (offset != kOutputResultFileBytes) {
        throw FormatError("internal result encoder size mismatch");
    }
}

[[nodiscard]] Bytes encode_result(std::uint64_t chart_id, ResultStatus status,
                                  const Counts& counts, std::span<const std::uint8_t> error_fields,
                                  const Digest& frame_begin, const Digest& frame_end,
                                  const Digest& result) {
    Bytes output(kOutputResultFileBytes);
    encode_result_into(std::span<std::uint8_t>(output.data(), output.size()), chart_id, status,
                       counts, error_fields, frame_begin, frame_end, result);
    return output;
}

// The payload is rendered directly into the worker's output buffer.
void fast_process(std::uint64_t chart_id,std::string_view text,
                  std::span<std::uint8_t,kernel::kEncodedChartBytes> out) {
    const FastChart& chart=maimoe::fast::parse_maidata(text);
    auto begin=out.subspan<kOutputFrameHeaderBytes,kOutputFramePayloadBytes>();
    auto end=out.subspan<kOutputFrameFileBytes+kOutputFrameHeaderBytes,kOutputFramePayloadBytes>();
    std::array<std::uint8_t,kSerializedStateBytes> state_bytes;
    ResultStatus status=ResultStatus::Valid;
    Counts counts{};
    Bytes fields;
    if(const auto* error=select_semantic_error(chart)) {
        status=ResultStatus::Error;fields=canonical_error_fields(*error);
        auto fill_error=[&](auto payload,FrameRole role) {
            Bytes input; input.reserve(64);
            append_lp32(input,"MaiMoeErrorFrame/v3");append_le(input,chart_id);
            input.push_back(static_cast<std::uint8_t>(role));append_lp32(input,fields);
            const auto digest=fast_sha256(input);
            for(std::size_t offset=0;offset<payload.size();offset+=digest.size())
                std::memcpy(payload.data()+offset,digest.data(),digest.size());
        };
        fill_error(begin,FrameRole::Begin);fill_error(end,FrameRole::End);
    } else {
        counts=chart.counts;
        const auto states=fast_evolve_endpoint_states(chart_id,chart);
        fast_render_output(chart_id,chart,states.frame_begin,kFirstSampleFrame,begin);
        fast_render_output(chart_id,chart,states.frame_end,kLastSampleFrame,end);
        serialize_state_into(states.frame_end,state_bytes);
    }
    const auto begin_digest=fast_sha256(begin),end_digest=fast_sha256(end);
    const auto result=status==ResultStatus::Error
        ? error_result_digest(chart_id,fields,begin_digest,end_digest)
        : valid_result_digest(chart_id,counts,state_bytes,begin_digest,end_digest);
    auto frame_header=[&](std::span<std::uint8_t> header,FrameRole role,const Digest& digest){
        std::size_t p=0;
        std::memcpy(header.data(),kFrameMagic.data(),8);p=8;
        append_le_into(header,p,static_cast<std::uint16_t>(kOutputFrameHeaderBytes));
        append_le_into(header,p,kOutputFrameWidth);append_le_into(header,p,kOutputFrameHeight);
        header[p++]=2;header[p++]=static_cast<std::uint8_t>(role);
        append_le_into(header,p,chart_id);header[p++]=static_cast<std::uint8_t>(status);
        header[p++]=0;append_le_into(header,p,std::uint16_t{0});
        append_le_into(header,p,kOutputFramePayloadBytes);append_digest_into(header,p,digest);
    };
    frame_header(out.first<kOutputFrameHeaderBytes>(),FrameRole::Begin,begin_digest);
    frame_header(out.subspan<kOutputFrameFileBytes,kOutputFrameHeaderBytes>(),FrameRole::End,end_digest);
    encode_result_into(out.last<kOutputResultFileBytes>(),chart_id,status,counts,fields,begin_digest,end_digest,result);
}
} // namespace maimoe::fast
namespace maimoe::kernel {
void process_chart(std::uint64_t chart_id,std::string_view text,std::span<std::uint8_t,kEncodedChartBytes> output) {
    fast::fast_process(chart_id,text,output);
}
}
