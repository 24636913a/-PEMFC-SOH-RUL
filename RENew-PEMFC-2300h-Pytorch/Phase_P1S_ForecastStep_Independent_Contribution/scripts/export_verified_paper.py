"""Export a new review-passed package. Exclusive writes; never replaces existing paper files."""
from common import *
DEST=ROOT/'REAL_Paper_File/P1S_ForecastStep_Independent_Contribution'
def export():
 config();assert load_local('outputs/INDEPENDENT_VERIFICATION.json')['PASS'];assert load_local('outputs/TEST_RESULTS.json')['PASS']
 review=load_local('outputs/FIGURE_VISUAL_REVIEW.json');assert review['PASS'] and len(review['figures'])==4
 for f,r in review['figures'].items():
  assert sha(P/f'figures/{f}.png')==r['png_sha256'] and sha(P/f'figures/{f}.pdf')==r['pdf_sha256']
 claims=P/'paper_candidates/CLAIM_BOUNDARIES.md';assert claims.stat().st_size>300
 assert not DEST.exists(),'Existing formal package must never be overwritten'
 mappings=[]
 for f in (P/'figures').iterdir():
  if f.suffix in ['.png','.pdf']:mappings.append((f,'figures/'+f.name))
 for name in ['SOH_METRICS.csv','POINT_PREDICTIONS.csv','POINTWISE_WINS.csv','PAIRWISE_COMPARISONS.csv','SEQUENTIAL_CONTRIBUTION.csv',
  'ERROR_DECOMPOSITION.csv','H3_VS_H1_ERROR_CANCELLATION.csv','P1S_PROXY_METRICS.csv','ORIGIN_MEASUREMENT_TIME_AND_IDENTITY_AUDIT.csv','H3_INCREMENT_CONCENTRATION.csv']:
  mappings.append((P/'outputs'/name,'source_csv/'+name))
 for src,rel in [(P/'scripts/render_paper_figures.py','scripts/render_paper_figures.py'),(P/'FIGURE_SOURCE_INDEX.csv','FIGURE_SOURCE_INDEX.csv'),
  (claims,'CLAIM_BOUNDARIES.md'),(P/'reports/00_预测步骤独立贡献结论.md','研究结论.md'),(P/'reports/02_代理误差与SOH误差抵消.md','误差抵消说明.md'),
  (P/'configs/FROZEN_PROTOCOL.json','verification/FROZEN_PROTOCOL.json'),(P/'configs/FROZEN_PROTOCOL.sha256','verification/FROZEN_PROTOCOL.sha256'),
  (P/'outputs/INDEPENDENT_VERIFICATION.json','verification/INDEPENDENT_VERIFICATION.json'),(P/'outputs/TEST_RESULTS.json','verification/TEST_RESULTS.json'),
  (P/'outputs/FIGURE_VISUAL_REVIEW.json','verification/FIGURE_VISUAL_REVIEW.json'),(P/'inputs/SOURCE_SHA256.csv','verification/SOURCE_SHA256.csv'),
  (P/'inputs/calibration.json','verification/FROZEN_CALIBRATION_PARAMETERS.json'),(P/'inputs/links.csv','source_csv/FROZEN_EXACT_LINKS.csv')]:mappings.append((src,rel))
 # Prepare all source bytes and identities before creating the destination.
 payload=[(s,r,s.read_bytes(),sha(s)) for s,r in mappings];DEST.mkdir(parents=True,exist_ok=False);receipt=[]
 for src,rel,content,h in payload:
  target=(DEST/rel).resolve();assert target.is_relative_to(DEST.resolve());target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:f.write(content)
  assert sha(target)==h
  receipt.append({'source':src.relative_to(ROOT).as_posix(),'paper_file':target.relative_to(ROOT).as_posix(),'sha256':h,'bytes':len(content)})
 readme='''# P1S预测步骤独立贡献：已核验论文候选包

本包只包含核验与测试通过、图审通过且带明确主张边界的真实成果，不代表独立确认性在线SOH研究。H1/H3共用3参考事件；未来实测P1S只用于事后误差分解。

在本包目录运行`python -B scripts/render_paper_figures.py`，从source_csv重绘所有PNG/PDF。完整数值核验、冻结输入及Python测试在研究主目录的Phase_P1S_ForecastStep_Independent_Contribution，README给出命令。来源CSV、绘图源码、源文件SHA与核验回执均保留。

可写和不可写的科学主张见CLAIM_BOUNDARIES.md。H3可作为后续Ridge预测候选，不能据三个事件和误差抵消宣布其已成为可靠SOH主方法。保留起点实测P1S+固定校准作为必要主对照。
'''
 with (DEST/'README.md').open('x',encoding='utf-8',newline='\n') as f:f.write(readme)
 with (DEST/'.gitattributes').open('x',encoding='utf-8',newline='\n') as f:f.write('* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n*.png binary\n*.pdf binary\n')
 manifest=[{'path':q.relative_to(DEST).as_posix(),'bytes':q.stat().st_size,'sha256':sha(q)} for q in sorted(DEST.rglob('*')) if q.is_file()]
 with (DEST/'PACKAGE_SHA256.csv').open('x',encoding='utf-8',newline='') as f:
  import csv
  writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);writer.writeheader();writer.writerows(manifest)
 cs('outputs/PAPER_COPY_SHA256.csv',pd.DataFrame(receipt));js('outputs/PAPER_EXPORT_RECEIPT.json',{'PASS':True,'destination':DEST.relative_to(ROOT).as_posix(),'copied_verified_files':len(receipt),'preexisting_files_overwritten':0,'new_package_only':True})
 print('EXPORTED_VERIFIED_PAPER_PACKAGE',len(receipt),'files')
if __name__=='__main__':export()
