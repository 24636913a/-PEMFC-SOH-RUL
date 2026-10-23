"""Literal ASCII spectrum conversion and a single predeclared diagnostic circuit."""
import re,datetime
import numpy as np
from scipy.optimize import least_squares
def header(text):
 h=text.split('Impedance samples')[0]
 def value(key):
  m=re.search(r'^'+re.escape(key)+r'[^:]*:\s*(.*)$',h,re.M);return m.group(1).strip() if m else 'UNKNOWN'
 info={k:value(k) for k in ['File','System','Potential','Current','Temperature','Time']}
 m=re.search(r'([A-Z]{3}),\s*(\d{2})\.(\d{4})',info['File']);date=datetime.datetime.strptime(''.join(m.groups()),'%b%d%Y').date() if m else None
 times=re.findall(r'\d{2}:\d{2}:\d{2}',info['Time']);info['calendar_start']=str(date)+' '+times[0] if date and times else 'UNKNOWN';info['calendar_end']=str(date)+' '+times[-1] if date and times else 'UNKNOWN'
 a=re.search(r'\(area/cm\^2\):\s*([\d.]+)',h);info['area_cm2']=float(a.group(1)) if a else np.nan
 c=re.search(r'([\d.]+)A,\s*Ampl:\s*([\d.]+)A',info['Current']);info['current_A']=float(c.group(1)) if c else np.nan;info['amplitude_A']=float(c.group(2)) if c else np.nan
 info['instrument_channels']='|'.join(re.findall(r'CH\d+',info['System']));return info
def parse(text):
 if text.startswith('version https://git-lfs.github.com/spec/v1'):raise ValueError('LFS_POINTER_NOT_A_SPECTRUM')
 lines=text.splitlines();idx=next(i for i,s in enumerate(lines) if s.strip().startswith('Number') and 'Frequency/Hz' in s)
 names=lines[idx].split();rows=[]
 for s in lines[idx+1:]:
  if not s.strip():continue
  vals=s.split()
  if len(vals)!=8:raise ValueError('ASCII_DATA_ROW_CARDINALITY')
  rows.append([float(x) for x in vals])
 data=np.array(rows);f=data[:,1]
 if 'impedance_R/Ohm' in names:
  z=data[:,2]+1j*data[:,3];fmt='RECTANGULAR_OHM'
 elif 'impedance/Ohm' in names and 'Phase/deg' in names:
  z=data[:,2]*np.exp(1j*np.deg2rad(data[:,3]));fmt='MAGNITUDE_PHASE_DEG'
 else:raise ValueError('UNKNOWN_IMPEDANCE_COLUMNS')
 n=re.search(r'Lines:\s*(\d+)',text);assert n and len(rows)==int(n.group(1))
 assert np.array_equal(data[:,0],np.arange(1,len(data)+1))
 return header(text),f,z,data[:,4],data[:,5],data[:,6],data[:,7],fmt,names
def split_scans(f):
 k=int(np.argmax(f));return [np.arange(k+1),np.arange(k,len(f))]
def crosses(f,z,q):
 out=[]
 for b,inds in enumerate(split_scans(f)):
  keep=inds[f[inds]>=1000]
  for a,c in zip(keep[:-1],keep[1:]):
   if z[a].imag*z[c].imag<0:
    R=float(z[a].real-z[a].imag*(z[c].real-z[a].real)/(z[c].imag-z[a].imag))
    out.append(dict(scan=b,i=int(a),k=int(c),f1=float(f[a]),f2=float(f[c]),intercept_ohm=R,significance_min=float(min(q[a],q[c])),
     status='APPARENT_INTERCEPT_ONLY_PARASITICS_UNCERTIFIED'))
 return out
def circuit(f,p):
 R0,R1,t1,R2,t2=np.exp(np.asarray(p)[[0,1,2,4,5]]);a1=p[3];a2=p[6];w=2*np.pi*np.asarray(f)
 return R0+R1/(1+(1j*w*t1)**a1)+R2/(1+(1j*w*t2)**a2)
LOW=np.array([np.log(1e-7),np.log(1e-7),np.log(1e-7),.1,np.log(1e-7),np.log(1e-7),.1])
HIGH=np.array([np.log(.01),np.log(.1),np.log(1e4),1.,np.log(.1),np.log(1e4),1.])
def fit(f,z):
 # Frozen published artifact exclusion: cell <=1026Hz, 0.1Hz minimum.
 inds=split_scans(f)[1];inds=inds[(f[inds]<=1026)&(f[inds]>=.1)]
 ff=f[inds];zz=z[inds]
 if len(ff)<15 or not np.isfinite(zz).all() or (zz.real<=0).any():return None
 scale=np.abs(zz)
 def residual(p):
  e=(circuit(ff,p)-zz)/scale;return np.r_[e.real,e.imag]
 r0=max(float(zz.real.min()*.8),1e-6);span=max(float(zz.real.max()-zz.real.min()),1e-5)
 outputs=[]
 for ta,tb in [(0.0005,.1),(.001,.2),(.003,.5)]:
  initial=np.array([np.log(r0),np.log(span*.5),np.log(ta),.8,np.log(span*.5),np.log(tb),.8])
  opt=least_squares(residual,initial,bounds=(LOW,HIGH),max_nfev=2000,ftol=1e-10,xtol=1e-10,gtol=1e-10)
  outputs.append(opt)
 best=min(outputs,key=lambda v:v.cost);s=np.linalg.svd(best.jac,compute_uv=False)
 summary=dict(points=len(ff),fmin=float(ff.min()),fmax=float(ff.max()),R0_effective_model_ohm=float(np.exp(best.x[0])),
  relative_complex_RMSE_percent=float(np.sqrt(np.mean(abs((circuit(ff,best.x)-zz)/scale)**2))*100),
  cost=float(best.cost),success=bool(best.success),nfev=int(sum(v.nfev for v in outputs)),
  multistart_R0_min=float(min(np.exp(v.x[0]) for v in outputs)),multistart_R0_max=float(max(np.exp(v.x[0]) for v in outputs)),
  jac_condition=float(s[0]/s[-1]) if s[-1]>0 else 1e300,jac_rank=int(np.linalg.matrix_rank(best.jac)),
  parameters=best.x.tolist(),singular_values=s.tolist(),active_bounds=best.active_mask.tolist(),
  role='MODEL_DEPENDENT_EFFECTIVE_SERIES_RESISTANCE_NOT_MEASURED_HFR',physical_parameter_certified=False)
 return summary,ff,zz,circuit(ff,best.x),residual,outputs
def profile(fit_result):
 summary,ff,zz,pred,residual,opts=fit_result;p=min(opts,key=lambda v:v.cost).x;out=[]
 for factor in [.5,.75,.9,1.,1.1,1.25,1.5]:
  fixed=p[0]+np.log(factor)
  def fun(x):return residual(np.r_[fixed,x])
  opt=least_squares(fun,p[1:],bounds=(LOW[1:],HIGH[1:]),max_nfev=1500,ftol=1e-10,xtol=1e-10,gtol=1e-10)
  out.append(dict(R0_fraction=factor,R0_ohm=float(np.exp(fixed)),cost=float(opt.cost),delta_cost=float(opt.cost-summary['cost']),profile_success=bool(opt.success),not_a_statistical_confidence_interval=True))
 return out
