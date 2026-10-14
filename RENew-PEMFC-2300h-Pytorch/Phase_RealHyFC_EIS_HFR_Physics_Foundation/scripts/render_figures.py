"""Real-data figures; never label model-derived R0 as measured HFR."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from common import P
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
def save(fig,name):
 fig.text(.02,-.015,'Source: allowlisted RealHyFC EIS ASCII / nominal polarization exports; exact files in SOURCE_CONTENT_MANIFEST.csv',fontsize=7,color='#555555')
 fig.savefig(P/'figures'/f'{name}.png',bbox_inches='tight');fig.savefig(P/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
def run():
 points=pd.read_csv(P/'outputs/EIS_NORMALIZED_POINTS.csv');meta=pd.read_csv(P/'outputs/EIS_MEASUREMENT_METADATA.csv');fit=pd.read_csv(P/'outputs/EQUIVALENT_CIRCUIT_DIAGNOSTICS.csv');fitpoints=pd.read_csv(P/'outputs/EQUIVALENT_CIRCUIT_POINT_RESIDUALS.csv')
 core=meta.loc[meta.role.eq('CORE_SPECTRUM')]
 selections=[('Cell_BoL_Jan2025','low_0.18','2025-01-08, approx0.18 A/cm²'),('Cell_BoL_Jan2025','high_1.8','2025-01-08, approx1.8 A/cm²'),
  ('Cell_1150nom_Apr2025','high_1.8','2025-04-24, nominal1150h'),('Spatial_BoL_Jan2025','high_1.8','2025-01-09, reference cell00'),
  ('Spatial_1150nom_Apr2025','high_1.8','2025-04-25, reference cell00'),('COMPOSITE','high_1.8','2025-04-24, five-cell reference')]
 groups=[]
 for family,load,title in selections:
  m=core.loc[(core.family.eq(family)&core.nominal_load.eq(load))] if family!='COMPOSITE' else core.loc[core.measurement_object.eq('FIVE_CELL_REFERENCE_COMPOSITE')]
  if family.startswith('Cell'):m=m.loc[m.measurement_object.eq('SINGLE_CELL_CHANNEL')]
  elif family.startswith('Spatial'):m=m.loc[m.export_index.eq(0)]
  groups.append((m,title))
 fig,axs=plt.subplots(2,3,figsize=(10.2,6.7))
 for ax,(m,title) in zip(axs.flat,groups):
  for r in m.itertuples():
   d=points.loc[points.source_path.eq(r.source_path)];ax.plot(d.Z_real_ohm*1000,-d.Z_imag_ohm*1000,lw=.8,alpha=.8)
  ax.axhline(0,color='#777777',lw=.6);ax.set_title(title,fontsize=9);ax.set_xlabel("Export Z' (mΩ)");ax.set_ylabel("Export -Z'' (mΩ)")
 fig.suptitle('Measured Nyquist spectra: separate objects/events; no ageing normalization',fontsize=11);fig.tight_layout();save(fig,'F1_cell_and_reference_Nyquist')
 fig,axs=plt.subplots(1,2,figsize=(9,3.8))
 for ax,family,date in zip(axs,['Spatial_BoL_Jan2025','Spatial_1150nom_Apr2025'],['2025-01-09','2025-04-25']):
  for r in core.loc[core.family.eq(family)&core.export_index.gt(0)].itertuples():
   d=points.loc[points.source_path.eq(r.source_path)];ax.plot(d.Z_real_ohm*1000,-d.Z_imag_ohm*1000,lw=.7,color=plt.cm.viridis((r.export_index-1)/15))
  ax.axhline(0,color='#777777',lw=.6);ax.set_title(f'{date}: exported CDDM channels01–16');ax.set_xlabel("Raw Z' (mΩ)");ax.set_ylabel("Raw -Z'' (mΩ)")
 fig.suptitle('Raw spatial channels: signs preserved; no zone-ASR conversion',fontsize=11);fig.tight_layout();save(fig,'F2_spatial_raw_Nyquist')
 fig,axs=plt.subplots(2,3,figsize=(10.3,6.7))
 for ax,(m,title) in zip(axs.flat,groups):
  r=m.iloc[0];d=points.loc[points.source_path.eq(r.source_path)];k=int(np.argmax(d.frequency_Hz.to_numpy()));d=d.iloc[k:]
  ax.loglog(d.frequency_Hz,d.magnitude_ohm*1000,color='#4C78A8',lw=1);ax.set_ylabel('|Z| (mΩ)',color='#4C78A8');ax.set_xlabel('Frequency (Hz)');ax.set_title(title,fontsize=8)
  twin=ax.twinx();twin.semilogx(d.frequency_Hz,d.phase_deg,color='#F58518',lw=.9);twin.set_ylabel('Raw phase (degrees)',color='#F58518');twin.spines['top'].set_visible(False)
 fig.suptitle('Bode plots of representative actual exports; original phase retained',fontsize=11);fig.tight_layout();save(fig,'F3_Bode_representative_exports')
 q=pd.read_csv(P/'outputs/EIS_DATA_QUALITY.csv');q=q.merge(core[['source_path','nominal_load','export_index']],on='source_path')
 fig,axs=plt.subplots(1,2,figsize=(9,3.3))
 for family,load,label in selections[:3]:
  d=q.loc[q.family.eq(family)&q.nominal_load.eq(load)&q.object.eq('SINGLE_CELL_CHANNEL')].sort_values('export_index')
  axs[0].plot(d.export_index,d.highest_frequency_real_ohm*1000,'o-',ms=3,lw=.8,label=label)
  axs[1].plot(d.export_index,d.highest_frequency_imag_ohm*1000,'o-',ms=3,lw=.8,label=label)
 for ax in axs:ax.axhline(0,color='#777777',lw=.6);ax.set_xlabel('Export cell index (not a verified U-channel map)')
 axs[0].set_ylabel("Measured5kHz Z' (mΩ)");axs[1].set_ylabel("Measured5kHz Z'' (mΩ)");axs[0].legend(fontsize=7)
 fig.suptitle('Observable high-frequency features: neither panel is HFR',fontsize=11);fig.tight_layout();save(fig,'F4_high_frequency_features_NOT_HFR')
 fig,axs=plt.subplots(1,2,figsize=(9,3.3))
 ref=fit.loc[fit.object.eq('REFERENCE_SINGLE_CELL')].sort_values('family',ascending=False)
 labels=['2025-01-09\nreference00','2025-04-25\nreference00']
 # Explicitly choose calendar order, not alphabetical stage order.
 ref=fit.loc[fit.object.eq('REFERENCE_SINGLE_CELL')].set_index('family').loc[['Spatial_BoL_Jan2025','Spatial_1150nom_Apr2025']]
 axs[0].bar(labels,ref.R0_effective_model_ohm*1000,color=['#4C78A8','#F58518']);axs[0].set_ylabel('Circuit R0 (mΩ)');axs[0].set_ylim(0,.22)
 for k,v in enumerate(ref.R0_effective_model_ohm*1000):axs[0].text(k,v+.005,f'{v:.6f}',ha='center',fontsize=8)
 axs[1].bar(labels,ref.ASR_effective_model_candidate_ohm_cm2,color=['#4C78A8','#F58518']);axs[1].set_ylabel('Model R0 ×280cm² (Ωcm²)');axs[1].set_ylim(0,.061)
 for k,v in enumerate(ref.ASR_effective_model_candidate_ohm_cm2):axs[1].text(k,v+.001,f'{v:.6f}',ha='center',fontsize=8)
 fig.suptitle('Model-dependent diagnostic candidates; not certified HFR/ASR or ageing gain',fontsize=10);fig.tight_layout();save(fig,'F5_model_R0_and_area_scaled_candidates')
 pol=pd.read_csv(P/'outputs/POLARIZATION_DESCRIPTIVE_DWELL_CURVES.csv');polmeta=pd.read_csv(P/'outputs/POLARIZATION_SOURCE_AND_CONDITIONS.csv').set_index('nominal_checkpoint');fig,axs=plt.subplots(1,2,figsize=(9.2,3.8))
 for nominal in ['BoL','300h','600h','850h','1150h','1500h','1750h']:
  d=pol.loc[pol.nominal_checkpoint.eq(nominal)];date=str(polmeta.loc[nominal,'first_calendar'])[:10];axs[0].plot(d.j_A_cm2,d.V_active10_mean,lw=.9,label=f'{nominal}: {date}')
 axs[0].set_xlabel('Actual dwell mean j (A/cm²)');axs[0].set_ylabel('Measured active10-cell mean (V)');axs[0].legend(ncol=2,fontsize=7);axs[0].set_title('Nominal polarization, all7 allowed references')
 for family,color,date in [('Spatial_BoL_Jan2025','#4C78A8','2025-01-09'),('Spatial_1150nom_Apr2025','#F58518','2025-04-25')]:
  R=ref.loc[family].ASR_effective_model_candidate_ohm_cm2;j=np.linspace(0,1.8,80);axs[1].plot(j,1000*j*R,color=color,label=date)
 axs[1].set_xlabel('Hypothetical comparable-state j (A/cm²)');axs[1].set_ylabel('Model series-component j×R0×A (mV)');axs[1].legend(fontsize=8);axs[1].set_title('Diagnostic component only; not full voltage/SOH')
 fig.suptitle('Different objects and dates: no quantitative ageing-loss attribution',fontsize=11);fig.tight_layout();save(fig,'F6_polarization_and_ohmic_component_DIAGNOSTIC')
 profiles=pd.read_csv(P/'outputs/R0_PROFILE_SENSITIVITY.csv');fig,axs=plt.subplots(1,2,figsize=(9,3.3))
 for source,d in profiles.groupby('source_path'):
  label='Spatial-ref ' if 'spatially' in source else ('Cell-low ' if '10A_' in source else 'Cell-high ')
  label+= 'Jan' if '/BoL' in source else 'Apr'
  axs[0].plot(d.R0_fraction,np.maximum(d.delta_cost,0),'o-',ms=3,lw=.8,label=label)
 axs[0].set_xlabel('Fixed R0 / fitted R0');axs[0].set_ylabel('Increase in weighted complex LS cost');axs[0].legend(fontsize=7);axs[0].set_title('Nuisance-optimized profiles; no statistical CI')
 axs[1].hist(fit.relative_complex_RMSE_percent,bins=10,color='#4C78A8',edgecolor='white');axs[1].set_xlabel('Relative complex residual RMSE (%)');axs[1].set_ylabel('Diagnostic circuit fits');axs[1].set_title('Residuals of33 fits; not prediction accuracy')
 fig.tight_layout();save(fig,'F7_parameter_profiles_and_fit_residuals')
 fig,ax=plt.subplots(figsize=(9.3,3.2))
 groups2=[('Nominal cellEIS',core.loc[core.family.str.startswith('Cell')]),('Nominal spatialEIS',core.loc[core.family.str.startswith('Spatial')]),('Fresh BoL sensitivity',meta.loc[meta.role.ne('CORE_SPECTRUM')])]
 for k,(label,d) in enumerate(groups2):
  times=pd.to_datetime(d.calendar_start);ax.scatter(times,np.full(len(times),k),s=12,color=['#4C78A8','#F58518','#777777'][k],alpha=.7)
 ax.set_yticks(range(3),[x[0] for x in groups2]);ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));ax.set_xlabel('Actual dates from spectrum headers');ax.set_title('BoL names describe different measurement families; serial identity incomplete')
 fig.tight_layout();save(fig,'F8_measurement_identity_calendar')
 print('8 real-data figures rendered')
if __name__=='__main__':run()
