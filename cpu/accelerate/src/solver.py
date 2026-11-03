"""Direct formula using compiled C and OpenMP; no precomputed input work."""
import ctypes
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np

# Capture the process allocation before loading libgomp: OMP_PROC_BIND may
# bind the main thread to one core when the library is initialized.
_available_threads = min(32, len(os.sched_getaffinity(0)))

_SOURCE = r"""
#include <math.h>
#include <stdint.h>
#include <omp.h>

#include <arm_neon.h>
#include <stdlib.h>
typedef float64x2_t V;
#define S(x) vdupq_n_f64(x)
static inline V vsin(V x) {
 V k=vrndnq_f64(x*S(0.31830988618379067154));
 V r=(x-k*S(3.141592653589793116))-k*S(1.2246467991473532072e-16);
 V z=r*r;
 V p=S(-1.0/1307674368000.0);
 p=p*z+S(1.0/6227020800.0);
 p=p*z-S(1.0/39916800.0);
 p=p*z+S(1.0/362880.0);
 p=p*z-S(1.0/5040.0);
 p=p*z+S(1.0/120.0);
 p=p*z-S(1.0/6.0);
 V y=r+r*z*p;
 uint64x2_t sign=vshlq_n_u64(vreinterpretq_u64_s64(vcvtq_s64_f64(k)),63);
 y=vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(y),sign));
 if (__builtin_expect(vmaxvq_f64(vabsq_f64(x))>1000.0,0)) {
  for (int j=0;j<2;j++) if (fabs(x[j])>1000.0) y[j]=sin(x[j]);
 }
 return y;
}
static inline V vexp(V x) {
 V n=vrndnq_f64(x*S(1.4426950408889634074));
 V r=(x-n*S(0.69314718036912381649))-n*S(1.9082149292705877e-10);
 V p=S(1.0/3628800.0);
 p=p*r+S(1.0/362880.0); p=p*r+S(1.0/40320.0);
 p=p*r+S(1.0/5040.0); p=p*r+S(1.0/720.0);
 p=p*r+S(1.0/120.0); p=p*r+S(1.0/24.0);
 p=p*r+S(1.0/6.0); p=p*r+S(0.5); p=p*r+S(1.0); p=p*r+S(1.0);
 int64x2_t e=vcvtq_s64_f64(n)+vdupq_n_s64(1023);
 V y=p*vreinterpretq_f64_s64(vshlq_n_s64(e,52));
 if (__builtin_expect(vmaxvq_f64(vabsq_f64(x))>700.0,0)) {
  for (int j=0;j<2;j++) if (fabs(x[j])>700.0) y[j]=exp(x[j]);
 }
 return y;
}

static inline V vsin_fast(V x) {
 V k=vrndnq_f64(x*S(0.31830988618379067154));
 V r=(x-k*S(3.141592653589793116))-k*S(1.2246467991473532072e-16);
 V z=r*r;
 V p=S(-1.0/1307674368000.0);
 p=p*z+S(1.0/6227020800.0);
 p=p*z-S(1.0/39916800.0);
 p=p*z+S(1.0/362880.0);
 p=p*z-S(1.0/5040.0);
 p=p*z+S(1.0/120.0);
 p=p*z-S(1.0/6.0);
 V y=r+r*z*p;
 uint64x2_t sign=vshlq_n_u64(vreinterpretq_u64_s64(vcvtq_s64_f64(k)),63);
 y=vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(y),sign));

 return y;
}
static inline V vexp_fast(V x) {
 V n=vrndnq_f64(x*S(1.4426950408889634074));
 V r=(x-n*S(0.69314718036912381649))-n*S(1.9082149292705877e-10);
 V p=S(1.0/3628800.0);
 p=p*r+S(1.0/362880.0); p=p*r+S(1.0/40320.0);
 p=p*r+S(1.0/5040.0); p=p*r+S(1.0/720.0);
 p=p*r+S(1.0/120.0); p=p*r+S(1.0/24.0);
 p=p*r+S(1.0/6.0); p=p*r+S(0.5); p=p*r+S(1.0); p=p*r+S(1.0);
 int64x2_t e=vcvtq_s64_f64(n)+vdupq_n_s64(1023);
 V y=p*vreinterpretq_f64_s64(vshlq_n_s64(e,52));

 return y;
}
static void field_safe(int64_t Q,int64_t C,int64_t D,const float*x,const float*mu_input,
  const float*w,const float*s,const float*b,const float*t,const float*v_input,float*out,int threads) {
 double *mu=malloc((size_t)C*D*sizeof(double)), *v=malloc((size_t)C*D*sizeof(double));
 #pragma omp parallel for num_threads(threads)
 for(int64_t i=0;i<C*D;i++) {mu[i]=mu_input[i];v[i]=v_input[i];}

  #pragma omp parallel for schedule(static) num_threads(threads)
  for (int64_t q=0;q<Q;q+=16) {
V xx0[D], total0=S(0);
for(int d=0;d<D;d++) xx0[d]=(V){x[(q+0<Q?q+0:q)*D+d],x[(q+1<Q?q+1:q)*D+d]};
V xx1[D], total1=S(0);
for(int d=0;d<D;d++) xx1[d]=(V){x[(q+2<Q?q+2:q)*D+d],x[(q+3<Q?q+3:q)*D+d]};
V xx2[D], total2=S(0);
for(int d=0;d<D;d++) xx2[d]=(V){x[(q+4<Q?q+4:q)*D+d],x[(q+5<Q?q+5:q)*D+d]};
V xx3[D], total3=S(0);
for(int d=0;d<D;d++) xx3[d]=(V){x[(q+6<Q?q+6:q)*D+d],x[(q+7<Q?q+7:q)*D+d]};
V xx4[D], total4=S(0);
for(int d=0;d<D;d++) xx4[d]=(V){x[(q+8<Q?q+8:q)*D+d],x[(q+9<Q?q+9:q)*D+d]};
V xx5[D], total5=S(0);
for(int d=0;d<D;d++) xx5[d]=(V){x[(q+10<Q?q+10:q)*D+d],x[(q+11<Q?q+11:q)*D+d]};
V xx6[D], total6=S(0);
for(int d=0;d<D;d++) xx6[d]=(V){x[(q+12<Q?q+12:q)*D+d],x[(q+13<Q?q+13:q)*D+d]};
V xx7[D], total7=S(0);
for(int d=0;d<D;d++) xx7[d]=(V){x[(q+14<Q?q+14:q)*D+d],x[(q+15<Q?q+15:q)*D+d]};
for(int64_t c=0;c<C;c++){
V r0=S(0),p0=S(0);
V r1=S(0),p1=S(0);
V r2=S(0),p2=S(0);
V r3=S(0),p3=S(0);
V r4=S(0),p4=S(0);
V r5=S(0),p5=S(0);
V r6=S(0),p6=S(0);
V r7=S(0),p7=S(0);
for(int d=0;d<D;d++){ V m=S(mu[c*D+d]), vv=S(v[c*D+d]);
V delta0=xx0[d]-m; r0+=delta0*delta0; p0+=xx0[d]*vv;
V delta1=xx1[d]-m; r1+=delta1*delta1; p1+=xx1[d]*vv;
V delta2=xx2[d]-m; r2+=delta2*delta2; p2+=xx2[d]*vv;
V delta3=xx3[d]-m; r3+=delta3*delta3; p3+=xx3[d]*vv;
V delta4=xx4[d]-m; r4+=delta4*delta4; p4+=xx4[d]*vv;
V delta5=xx5[d]-m; r5+=delta5*delta5; p5+=xx5[d]*vv;
V delta6=xx6[d]-m; r6+=delta6*delta6; p6+=xx6[d]*vv;
V delta7=xx7[d]-m; r7+=delta7*delta7; p7+=xx7[d]*vv;
}
V e0=-S(s[c])*r0, a0=S(t[c])*p0;
total0+=S(w[c])*vexp(e0)+S(b[c])*vsin(a0);
V e1=-S(s[c])*r1, a1=S(t[c])*p1;
total1+=S(w[c])*vexp(e1)+S(b[c])*vsin(a1);
V e2=-S(s[c])*r2, a2=S(t[c])*p2;
total2+=S(w[c])*vexp(e2)+S(b[c])*vsin(a2);
V e3=-S(s[c])*r3, a3=S(t[c])*p3;
total3+=S(w[c])*vexp(e3)+S(b[c])*vsin(a3);
V e4=-S(s[c])*r4, a4=S(t[c])*p4;
total4+=S(w[c])*vexp(e4)+S(b[c])*vsin(a4);
V e5=-S(s[c])*r5, a5=S(t[c])*p5;
total5+=S(w[c])*vexp(e5)+S(b[c])*vsin(a5);
V e6=-S(s[c])*r6, a6=S(t[c])*p6;
total6+=S(w[c])*vexp(e6)+S(b[c])*vsin(a6);
V e7=-S(s[c])*r7, a7=S(t[c])*p7;
total7+=S(w[c])*vexp(e7)+S(b[c])*vsin(a7);
}
if(q+0<Q) out[q+0]=total0[0]; if(q+1<Q) out[q+1]=total0[1];
if(q+2<Q) out[q+2]=total1[0]; if(q+3<Q) out[q+3]=total1[1];
if(q+4<Q) out[q+4]=total2[0]; if(q+5<Q) out[q+5]=total2[1];
if(q+6<Q) out[q+6]=total3[0]; if(q+7<Q) out[q+7]=total3[1];
if(q+8<Q) out[q+8]=total4[0]; if(q+9<Q) out[q+9]=total4[1];
if(q+10<Q) out[q+10]=total5[0]; if(q+11<Q) out[q+11]=total5[1];
if(q+12<Q) out[q+12]=total6[0]; if(q+13<Q) out[q+13]=total6[1];
if(q+14<Q) out[q+14]=total7[0]; if(q+15<Q) out[q+15]=total7[1];
}free(mu);free(v); }
static void field_fast(int64_t Q,int64_t C,int64_t D,const float*x,const float*mu_input,
  const float*w,const float*s,const float*b,const float*t,const float*v_input,float*out,int threads) {
 double *mu=malloc((size_t)C*D*sizeof(double)), *v=malloc((size_t)C*D*sizeof(double));
 #pragma omp parallel for num_threads(threads)
 for(int64_t i=0;i<C*D;i++) {mu[i]=mu_input[i];v[i]=v_input[i];}

  #pragma omp parallel for schedule(static) num_threads(threads)
  for (int64_t q=0;q<Q;q+=16) {
V xx0[D], total0=S(0);
for(int d=0;d<D;d++) xx0[d]=(V){x[(q+0<Q?q+0:q)*D+d],x[(q+1<Q?q+1:q)*D+d]};
V xx1[D], total1=S(0);
for(int d=0;d<D;d++) xx1[d]=(V){x[(q+2<Q?q+2:q)*D+d],x[(q+3<Q?q+3:q)*D+d]};
V xx2[D], total2=S(0);
for(int d=0;d<D;d++) xx2[d]=(V){x[(q+4<Q?q+4:q)*D+d],x[(q+5<Q?q+5:q)*D+d]};
V xx3[D], total3=S(0);
for(int d=0;d<D;d++) xx3[d]=(V){x[(q+6<Q?q+6:q)*D+d],x[(q+7<Q?q+7:q)*D+d]};
V xx4[D], total4=S(0);
for(int d=0;d<D;d++) xx4[d]=(V){x[(q+8<Q?q+8:q)*D+d],x[(q+9<Q?q+9:q)*D+d]};
V xx5[D], total5=S(0);
for(int d=0;d<D;d++) xx5[d]=(V){x[(q+10<Q?q+10:q)*D+d],x[(q+11<Q?q+11:q)*D+d]};
V xx6[D], total6=S(0);
for(int d=0;d<D;d++) xx6[d]=(V){x[(q+12<Q?q+12:q)*D+d],x[(q+13<Q?q+13:q)*D+d]};
V xx7[D], total7=S(0);
for(int d=0;d<D;d++) xx7[d]=(V){x[(q+14<Q?q+14:q)*D+d],x[(q+15<Q?q+15:q)*D+d]};
for(int64_t c=0;c<C;c++){
V r0=S(0),p0=S(0);
V r1=S(0),p1=S(0);
V r2=S(0),p2=S(0);
V r3=S(0),p3=S(0);
V r4=S(0),p4=S(0);
V r5=S(0),p5=S(0);
V r6=S(0),p6=S(0);
V r7=S(0),p7=S(0);
for(int d=0;d<D;d++){ V m=S(mu[c*D+d]), vv=S(v[c*D+d]);
V delta0=xx0[d]-m; r0+=delta0*delta0; p0+=xx0[d]*vv;
V delta1=xx1[d]-m; r1+=delta1*delta1; p1+=xx1[d]*vv;
V delta2=xx2[d]-m; r2+=delta2*delta2; p2+=xx2[d]*vv;
V delta3=xx3[d]-m; r3+=delta3*delta3; p3+=xx3[d]*vv;
V delta4=xx4[d]-m; r4+=delta4*delta4; p4+=xx4[d]*vv;
V delta5=xx5[d]-m; r5+=delta5*delta5; p5+=xx5[d]*vv;
V delta6=xx6[d]-m; r6+=delta6*delta6; p6+=xx6[d]*vv;
V delta7=xx7[d]-m; r7+=delta7*delta7; p7+=xx7[d]*vv;
}
V e0=-S(s[c])*r0, a0=S(t[c])*p0;
total0+=S(w[c])*vexp_fast(e0)+S(b[c])*vsin_fast(a0);
V e1=-S(s[c])*r1, a1=S(t[c])*p1;
total1+=S(w[c])*vexp_fast(e1)+S(b[c])*vsin_fast(a1);
V e2=-S(s[c])*r2, a2=S(t[c])*p2;
total2+=S(w[c])*vexp_fast(e2)+S(b[c])*vsin_fast(a2);
V e3=-S(s[c])*r3, a3=S(t[c])*p3;
total3+=S(w[c])*vexp_fast(e3)+S(b[c])*vsin_fast(a3);
V e4=-S(s[c])*r4, a4=S(t[c])*p4;
total4+=S(w[c])*vexp_fast(e4)+S(b[c])*vsin_fast(a4);
V e5=-S(s[c])*r5, a5=S(t[c])*p5;
total5+=S(w[c])*vexp_fast(e5)+S(b[c])*vsin_fast(a5);
V e6=-S(s[c])*r6, a6=S(t[c])*p6;
total6+=S(w[c])*vexp_fast(e6)+S(b[c])*vsin_fast(a6);
V e7=-S(s[c])*r7, a7=S(t[c])*p7;
total7+=S(w[c])*vexp_fast(e7)+S(b[c])*vsin_fast(a7);
}
if(q+0<Q) out[q+0]=total0[0]; if(q+1<Q) out[q+1]=total0[1];
if(q+2<Q) out[q+2]=total1[0]; if(q+3<Q) out[q+3]=total1[1];
if(q+4<Q) out[q+4]=total2[0]; if(q+5<Q) out[q+5]=total2[1];
if(q+6<Q) out[q+6]=total3[0]; if(q+7<Q) out[q+7]=total3[1];
if(q+8<Q) out[q+8]=total4[0]; if(q+9<Q) out[q+9]=total4[1];
if(q+10<Q) out[q+10]=total5[0]; if(q+11<Q) out[q+11]=total5[1];
if(q+12<Q) out[q+12]=total6[0]; if(q+13<Q) out[q+13]=total6[1];
if(q+14<Q) out[q+14]=total7[0]; if(q+15<Q) out[q+15]=total7[1];
}free(mu);free(v); }
void field(int64_t Q,int64_t C,int64_t D,const float*x,const float*mu,
 const float*w,const float*s,const float*b,const float*t,const float*v,float*out,int threads) {
 double nx=0,nm=0,tv=0,sm=0;
 #pragma omp parallel for num_threads(threads) reduction(max:nx)
 for(int64_t q=0;q<Q;q++) {double a=0;for(int d=0;d<D;d++){double y=x[q*D+d];a+=y*y;}if(a>nx)nx=a;}
 #pragma omp parallel for num_threads(threads) reduction(max:nm,tv,sm)
 for(int64_t c=0;c<C;c++) {
  double a=0,b=0;
  for(int d=0;d<D;d++){double y=mu[c*D+d],z=v[c*D+d];a+=y*y;b+=z*z;}
  if(a>nm)nm=a; b*=((double)t[c]*t[c]); if(b>tv)tv=b;
  double ss=fabs((double)s[c]);if(ss>sm)sm=ss;
 }
 double normsum=sqrt(nx)+sqrt(nm);
 if(nx*tv<900.0*900.0 && sm*normsum*normsum<690.0)
  field_fast(Q,C,D,x,mu,w,s,b,t,v,out,threads);
 else field_safe(Q,C,D,x,mu,w,s,b,t,v,out,threads);
}

"""
_build = tempfile.TemporaryDirectory(prefix="accelerate-c-")
_c = Path(_build.name) / "field.c"
_so = Path(_build.name) / "field.so"
_c.write_text(_SOURCE)
subprocess.run([os.environ.get("CC", "gcc"), "-O3", "-fopenmp", "-mcpu=native",
                "-ffp-contract=fast", "-shared", "-fPIC", str(_c),
                "-o", str(_so), "-lm"], check=True, capture_output=True)
_lib = ctypes.CDLL(str(_so))
_ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
_lib.field.argtypes = [ctypes.c_int64]*3 + [_ptr]*8 + [ctypes.c_int]
_lib.field.restype = None

def compute_field(points, centers, weights, scales, bias, trig_scale, trig_vec):
    q, d = points.shape
    c = centers.shape[0]
    if centers.shape != (c,d) or trig_vec.shape != (c,d):
        raise ValueError("inconsistent matrix shapes")
    if any(a.shape != (c,) for a in (weights, scales, bias, trig_scale)):
        raise ValueError("inconsistent vector shapes")
    output = np.empty(q, dtype=np.float32)
    threads = min(_available_threads, max(1,q))
    _lib.field(q,c,d,points,centers,weights,scales,bias,trig_scale,trig_vec,output,threads)
    return output
