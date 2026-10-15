"""No model scores: qualify cached platforms, freeze both tasks and all settings."""
from common import *
TRAIN_CURRENTS=[100.8,151.2,218.4,302.4,403.2,504.]
HOLDOUT_CURRENTS=[126.,201.6,252.,352.8,453.6]
COND=['T.Si.A [°C]_mean','T.Si.C [°C]_mean','DPT.Si.A [°C]_mean','DPT.Si.C [°C]_mean','p.So.A [kPag]_mean','p.So.C [kPag]_mean','Stoic.S.A [-]_mean','Stoic.S.C [-]_mean']
TOL=[.5,.5,.5,.5,2.,2.,.1,.1]
def run():
 assert not (P/'configs/FROZEN_PROTOCOL.json').exists(),'Already frozen; never resplit after scores'
 d=rd('dwells');sl=rd('slopes');meta=rd('metadata').set_index('checkpoint');assert json.loads((ROOT/SOURCES['verification']).read_text())['status']=='PASS'
 candidate=d[d.window.eq('middle50')&d.branch.isin(['ascending','descending'])&d.set_j_A_cm2.between(.36-1e-10,1.8+1e-10)].copy();assert len(candidate)==154
 good=candidate[candidate.stable_domain].copy();assert len(good)==105 and len(sl)==402 and not sl.monotonic_violation.any()
 assert (abs(good.j_A_cm2-good.set_j_A_cm2)<=.005).all()
 good['sample_id']=[f'{r.checkpoint}_{r.branch}_{int(round(r.set_current_A*10))}' for r in good.itertuples()];good['j_design_A_cm2']=good.set_current_A/280;good['direction_ascending']=good.branch.eq('ascending').astype(int);good['true_V']=good.V_V
 cs('data_contracts/ALL_154_PLATFORM_QUALIFICATION.csv',candidate);cs('data_contracts/LEGAL_105_PLATFORMS.csv',good)
 a=[];excluded=[];Atrain=0;Aval=0
 for (k,b),g in good.groupby(['checkpoint','branch'],sort=False):
  tr=g[np.isclose(g.set_current_A.to_numpy()[:,None],TRAIN_CURRENTS,atol=1e-8).any(axis=1)];v=g[np.isclose(g.set_current_A.to_numpy()[:,None],HOLDOUT_CURRENTS,atol=1e-8).any(axis=1)];anchor=tr[abs(tr.set_current_A-100.8)<1e-8];assert len(anchor)==1
  for r in g.to_dict('records'):
   role='train' if r['sample_id'] in set(tr.sample_id) else 'validation'
   if role=='validation' and not tr.j_design_A_cm2.min()<r['j_design_A_cm2']<tr.j_design_A_cm2.max():excluded.append({'task':'A','sample_id':r['sample_id'],'reason':'HOLDOUT_NOT_BRACKETED_BY_SAME_EVENT_BRANCH_TRAIN'});continue
   a.append(r|{'task':'A','fold':'within_curve','role':role,'context':float(anchor.V_V.iloc[0]),'context_source_id':anchor.sample_id.iloc[0],'context_definition':'Observed training-platform V at commandedj0.36 in same checkpoint/branch; reconstruction only','origin_calendar':meta.loc[k,'end_calendar'],'target_event':k,'domain_low':float(tr.j_design_A_cm2.min()),'domain_high':float(tr.j_design_A_cm2.max())})
 af=pd.DataFrame(a);cs('data_contracts/A_SAMPLES.csv',af)
 # First measured causal normalcondition P1S after previous polarization completion.
 ids=rd('identity');meas=rd('measurement');meas=meas[(meas.stable_window=='middle_50')&meas.cycle_interval_id.isin(ids.cycle_interval_id)].copy();meas['calendar']=pd.to_datetime(meas.measurement_end_datetime)
 normal=meas[meas.anode_inlet_temperature_C.between(74,76)&meas.anode_inlet_RH_pct.between(39,41)].sort_values('calendar');events=[]
 for prev,k in zip(EVENTS[:-1],EVENTS[1:]):
  pool=normal[(normal.calendar>pd.Timestamp(meta.loc[prev,'end_calendar']))&(normal.calendar<pd.Timestamp(meta.loc[k,'start_calendar']))]
  if len(pool):
   o=pool.iloc[0];events.append({'checkpoint':k,'previous_checkpoint':prev,'origin_calendar':str(o.calendar),'context':float(o.P1S_operating_voltage_retention),'context_source_id':o.cycle_interval_id,'previous_curve_end':meta.loc[prev,'end_calendar'],'target_first_calendar':meta.loc[k,'start_calendar'],'target_last_calendar':meta.loc[k,'end_calendar'],'calendar_lead_h_AUDIT_ONLY':(pd.Timestamp(meta.loc[k,'start_calendar'])-o.calendar).total_seconds()/3600,'available':True})
  else:events.append({'checkpoint':k,'previous_checkpoint':prev,'available':False})
 ev=pd.DataFrame(events);cs('data_contracts/B_EVENT_ORIGIN_IDENTITY.csv',ev);B=[];gates=[]
 for target in ['1150h','1500h','1750h']:
  past=[k for k in EVENTS[1:] if EVENTS.index(k)<EVENTS.index(target)];fold='before_'+target
  sources=ev[ev.checkpoint.isin(past+[target])];okay=bool(sources.available.all() and len(sources)==len(past)+1)
  part=[];bounds={}
  if okay:
   for branch in ['ascending','descending']:
    groups=[good[(good.checkpoint==k)&(good.branch==branch)] for k in past];lo=max(g.j_design_A_cm2.min() for g in groups);hi=min(g.j_design_A_cm2.max() for g in groups);bounds[branch]=[float(lo),float(hi)]
    for k in past+[target]:
     event=sources[sources.checkpoint==k].iloc[0];role='validation' if k==target else 'train'
     for r in good[(good.checkpoint==k)&(good.branch==branch)].to_dict('records'):
      reason=[]
      if not lo-1e-10<=r['j_design_A_cm2']<=hi+1e-10:reason.append('OUTSIDE_TRAIN_ONLY_COMMON_CURRENT_DOMAIN')
      if role=='validation':
       matched=good[good.checkpoint.isin(past)&good.branch.eq(branch)&(abs(good.set_current_A-r['set_current_A'])<1e-8)]
       if not len(matched):reason.append('NO_PAST_MATCHED_SETPOINT')
       elif any(r[c]<matched[c].min()-tol or r[c]>matched[c].max()+tol for c,tol in zip(COND,TOL)):reason.append('FUTURE_CONDITION_OUTSIDE_PREDECLARED_REFERENCE_SUPPORT')
      if reason:excluded.append({'task':'B','fold':fold,'sample_id':r['sample_id'],'role':role,'reason':'|'.join(reason)});continue
      part.append(r|{'task':'B','fold':fold,'role':role,'context':float(event.context),'context_source_id':event.context_source_id,'context_definition':'First causal normalcondition P1S after priorcurve completion; no target-event measurements in inputs','origin_calendar':event.origin_calendar,'target_event':k,'domain_low':float(lo),'domain_high':float(hi)})
   temp=pd.DataFrame(part);train=temp[temp.role.eq('train')];val=temp[temp.role.eq('validation')]
   okay=bool(len(train)>=12 and train.checkpoint.nunique()>=3 and len(val)>=4 and val.branch.nunique()==2 and pd.to_datetime(train.end_calendar).max()<pd.Timestamp(sources[sources.checkpoint==target].origin_calendar.iloc[0]))
   if okay:B+=part
  gates.append({'task':'B','fold':fold,'target_event':target,'qualified':okay,'Train_events':'|'.join(past),'Train_n':sum(r['role']=='train' for r in part),'Validation_n':sum(r['role']=='validation' for r in part),'domain_by_branch':str(bounds),'status':'QUALIFIED_CONDITIONAL_NOMINAL_PROTOCOL_REPLAY' if okay else 'FUTURE_POLARIZATION_FORECAST_NOT_EVALUABLE','strict_online_release_certified':False})
 bf=pd.DataFrame(B);cs('data_contracts/B_SAMPLES.csv',bf);cs('data_contracts/EXCLUDED_PLATFORM_RECORDS.csv',pd.DataFrame(excluded));cs('outputs/B_TIME_QUALIFICATION.csv',pd.DataFrame(gates))
 allf=pd.concat([af,bf],ignore_index=True);cs('data_contracts/ALL_LEGAL_SAMPLES.csv',allf)
 # Physically unlabeled probes use Train-only current bounds and Train-only context values.
 probes=[];scalers=[]
 for (task,fold),g in allf.groupby(['task','fold'],sort=False):
  tr=g[g.role.eq('train')];mu=tr[FEATURES].mean().to_numpy();sd=tr[FEATURES].std(ddof=0).to_numpy(copy=True);sd[sd==0]=1.;ym=float(tr.true_V.mean());ys=float(tr.true_V.std(ddof=0));assert ys>0
  scalers.append({'task':task,'fold':fold,'x_mean':mu.tolist(),'x_scale':sd.tolist(),'y_mean':ym,'y_scale':ys,'training_sample_ids':tr.sample_id.tolist()})
  for (event,branch),z in tr.groupby(['checkpoint','branch']):
   lo=float(z.domain_low.iloc[0]);hi=float(z.domain_high.iloc[0]);context=float(z.context.iloc[0]);direction=int(z.direction_ascending.iloc[0])
   for j in np.linspace(lo,hi,17):probes.append({'task':task,'fold':fold,'training_context_event':event,'branch':branch,'j_design_A_cm2':float(j),'direction_ascending':direction,'context':context,'context_source_id':z.context_source_id.iloc[0],'not_real_measurement':True,'no_voltage_label':True})
 cs('data_contracts/UNLABELED_TRAIN_PHYSICS_PROBES.csv',pd.DataFrame(probes));js('data_contracts/TRAIN_ONLY_SCALERS.json',scalers)
 cfg={'base_commit':'952f2f462f1834c4faaefc29c9dce4f8eb23c4c8','source_sha256':{k:sha(ROOT/v) for k,v in SOURCES.items()},'source_paths':SOURCES,'code_sha256':{q.relative_to(P).as_posix():sha(q) for q in (P/'scripts').glob('*.py') if q.name in ['common.py','prepare.py','network.py','train.py']},'data_contract_sha256':{q.relative_to(P).as_posix():sha(q) for q in (P/'data_contracts').glob('*')},
 'CPU_only':True,'features':FEATURES,'input_j':'Commanded current/280cm2; scheduled target setpoints known, never futureactualcurrent. Actualj disagreement<=0.005 checked. Continuousj derivative uses physicalcommandeddensity units.',
 'conditions':'No futureactualenvironment input; target is nominalprotocol conditional mean voltage; observedT/dewpoint/pressure/stoic only predeclared eligibility/supportaudit. Not a condition-normalized universalvoltage model.',
 'A':'Within measuredcheckpoint reconstruction. GlobalMLP j,direction,knowntraininganchorV atj0.36. Exactplatformholdout allwindows; origins atmeasurementcompletion, not futureforecast. Require heldpoint bracketed sameeventbranchTrain.',
 'A_training_setpoints_A':TRAIN_CURRENTS,'A_holdout_setpoints_A':HOLDOUT_CURRENTS,
 'B':'Next scheduled Developmentevent backtest. Expandingpast300/600/850 etc; zfirstnormalP1S after priorcurveend before target. Prescribedtargetj/direction, no futureactualconditions/targetcalendar/actuallead/Rtargets. ModelTrainlabelsend before validationorigin. Measurementcompletion assumption, publicationtime unknown, not strictonlineSOH.',
 'B_targets':['1150h','1500h','1750h'],'B_reference_support_tolerances':dict(zip(COND,TOL)),'B_failure_rule':'If origin/state/comparableTraincurrent or>=4Valpoints/2branches unavailable -> no Btraining, FUTURE_POLARIZATION_FORECAST_NOT_EVALUABLE',
 'architecture':{'hidden':[32,32],'activation':'SiLU','output':1,'parameters':1217,'dtype':'float64'},'seeds':[42,52,62],'epochs':2500,'optimizer':'Adam','learning_rate':.001,'weight_decay':0,'batch':'full','early_stopping':'NONE; fixedterminalepoch; no Validation-basedselection','lambda':1.,
 'loss_data':'mean((Vpred-Vtrue)/Train_y_scale)^2','loss_phys':'mean(relu(dV/dj * Train_j_scale/Train_y_scale)^2). Equal to mean(relu(dnormalizedV/dnormalizedj)^2); rawloss units(ohm cm2)^2 also recorded.',
 'autograd':'create_graph=True onnormalizedj; restore dV/dj=y_scale/x_j_scale*dnetwork/dx_j; gradient through networkparameters. Context/direction heldfixed; no time-monotonicconstraint.',
 'probes':'17uniformj per training eventbranch withinTrain-only support/context; unlabeled, never newmeasuredlabels; A/B shareexactprobes and architectureinitialweights and optimizersteps.',
 'training_monitor':'Everyepoch realLphys, violationrate, dVdjmax, actualdata/physicalparametergradientL2, weightedgradientratio. True inactive loss/gradient may mathematically be0; invalid valuesNA, never fabricated0.',
 'primary':'TaskA checkpoint-equal MAE acrossalllegalheldplatforms, seeds paired. B event-equal acrossqualified3events separately. No pvalues/IIDlifetimes.',
 'success':'For eachtask separately: >=2/3seedreductions in violationrate or violationRMS by>=50% relative (baseline>1e-8); seedmean MAE andRMSE notworse (1e-9mV tolerance), MAE gain>=2% and>=2/3seedMAE wins; no eventseedmeanMAE >20%worse. Allrequired, else no clearincrement. These are fixedexploratory decisioncriteria, not significance.',
 'inactive':'If Btraining never has Lphys>1e-16 AND parametergradientnorm>1e-12, PHYSICS_CONSTRAINT_INACTIVE. No lambda/probe/modelchanges afterscores.',
 'interfaces':'No TFT/SOHtraining. Reconstructioncontext unavailable beforefutureevent; Bconditionalnominalvoltage not referenceSOH label. No certification of SOH/RUL orTFTincrement.',
 'stop':'Each runID once, checkpointresume technicalinterruptions only. No rescuegrid/newseed/lambda/backbone. Exportverifiedexploratoryartifacts only, preserveTFTandHseries.'}
 js('configs/FROZEN_PROTOCOL.json',cfg);md('configs/FROZEN_PROTOCOL.sha256',sha(P/'configs/FROZEN_PROTOCOL.json'));cs('data_contracts/SOURCE_SHA256.csv',pd.DataFrame([{'key':k,'path':v,'SHA256':cfg['source_sha256'][k]} for k,v in SOURCES.items()]))
 for folder in ['models','figures','reports','tests']:out(folder+'/.keep').touch(exist_ok=True)
 md('.gitattributes','* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n*.png binary\n*.pdf binary\n*.pt binary');md('.gitignore','__pycache__/\n*.pyc\noutputs/GIT_RECEIPT.local.json')
 print('FROZEN',sha(P/'configs/FROZEN_PROTOCOL.json'));print(allf.groupby(['task','fold','role']).size().to_string());print(pd.DataFrame(gates).to_string(index=False))
if __name__=='__main__':firewall();run()
