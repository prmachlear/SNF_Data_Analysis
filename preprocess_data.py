INTENSIVE_CARE_INDICATOR=[0,1,2,3,4,5,6,7,8,9]
CORONARY_CARE_INDICATOR=[0,1,2,3,4,9]
SPECIAL_UNIT_CHARACTER_CODE=['M','R','S','T','U','W','Y','Z']
PRIMARY_PAYER_CODE=['A','B','C','D','E','F','G','H','I','J','Z']
SOURCE_OF_ADMISSION=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F',0,1,2,3,4,5,6,7,8,9]
HRR_PRTCPNT_IND=[0,1,2]
POA_DIAGNOSIS_INDICATOR_=['X','Y','Z','U','W','N','0','1',0,1]
POA_DIAGNOSIS_E_INDICATOR_1=['X','Y','Z','U','W','N','0','1',0,1]
HMO_PAID_INDICATOR=['0','1','2','4','A','B','C']
DRG_CODE=[37, 38, 40, 41, 42, 52, 53, 54, 55, 56, 57, 58, 59, 60, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 121, 122, 123, 124, 125, 133, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 166, 167, 168, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 228, 242, 252, 253, 254, 255, 256, 259, 264, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 330, 348, 356, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 409, 416, 417, 418, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 460, 462, 463, 464, 465, 467, 470, 474, 477, 478, 481, 482, 483, 492, 493, 494, 500, 501, 503, 504, 505, 513, 515, 516, 517, 519, 520, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 570, 571, 572, 574, 578, 579, 580, 581, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 617, 622, 623, 624, 628, 629, 637, 638, 639, 640, 641, 642, 643, 644, 645, 662, 669, 673, 674, 675, 682, 683, 684, 685, 686, 687, 688, 689, 690, 693, 694, 695, 696, 697, 698, 699, 700, 713, 722, 723, 724, 725, 726, 727, 728, 729, 730, 746, 747, 749, 750, 754, 755, 756, 757, 758, 759, 760, 761, 774, 775, 776, 778, 779, 781, 782, 789, 790, 791, 793, 794, 808, 809, 810, 811, 812, 813, 814, 815, 816, 824, 829, 834, 835, 836, 838, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 853, 854, 855, 856, 857, 858, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 876, 880, 881, 882, 883, 884, 885, 886, 887, 894, 895, 896, 897, 901, 902, 903, 904, 905, 907, 908, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 929, 933, 934, 935, 939, 940, 941, 945, 946, 947, 948, 949, 950, 951, 963, 964, 965, 974, 975, 976, 977, 981, 982, 983, 987, 988, 989, 998, 999]
INTERMEDIARY_NUMBER=[3201, 14211, 7301, 8201, 11401, 5901, 12301, 4111, 9101, 13201, 3601, 2201, 5401, 3101, 14111, 1311, 7201, 8101, 11301, 12201, 3501, 13101, 4911, 14511, 2101, 5301, 2301, 6201, 4411, 1211, 7101, 10301, 11201, 12101, 3401, 14411, 5201, 6101, 12501, 1111, 4311, 10201, 15201, 2401, 12901, 3301, 14311, 5101, 11501, 12401, 6001, 4211, 9201, 10101, 1911, 11004, 15101]
NCH_CLAIM_TYPE_CODE=[10,20,30,40,50,60,61,62,63,64,71,72,81,82]
AGE=[0,1,2,3,4,5,6,7,8,9]
SEX=[0,1,2] #unknown, male, female
RACE=[0,1,2,3,4,5,6]
STATE=[int(x) for x in range(1,75)]+[80,97,98,99]
PPS_INDICATOR=[0,1,2,3]
PHARMACY_INDICATOR=[0,1,2,3,4,5]
TRANSPLANT_INDICATOR=[0,2,7]
RADIOLOGY_INDICATOR_=[0,1]
OUTPATIENT_SERVICES_INDICATOR=[0,1,2,3]
SURGERY_INDICATOR=[0,1]
DISCHARGE_DESTINATION=[1,2,3,4,5,6,7,8,9,20,30,40,41,42,43,50,51,61,62,63,64,65,66,70,71,72]
TYPE_OF_ADMISSION=[0,1,2,3,4,5,6,7,8,9]
INFRMTL_ENCTR_IND=['Y','N']

def getPaths(pathfile_dir,pathfile):
	pathdict={}
	with open(pathfile_dir+pathfile) as f:
		for line in f:
			y=line.strip('\n')
			x=y.split('=')
			pathdict[x[0]]=x[1]
	return pathdict

def generateDict(pathcodes,diseasecodesfile,proceduresfile):
	diseases9=[]
	procedures9=[]
	with open(pathcodes+diseasecodesfile,encoding ='ISO-8859-1') as f:
		for line in f:
			diseases9.append(line.split(' ')[0])
	with open(pathcodes+proceduresfile,encoding ='ISO-8859-1') as f:
		for line in f:
			procedures9.append(line.split(' ')[0]) 
	colList1=['HMO_PAID_INDICATOR','INTENSIVE_CARE_INDICATOR','POA_DIAGNOSIS_INDICATOR_','CORONARY_CARE_INDICATOR','SPECIAL_UNIT_CHARACTER_CODE','SOURCE_OF_ADMISSION','DRG_CODE','INTERMEDIARY_NUMBER','NCH_CLAIM_TYPE_CODE','AGE','SEX','RACE','STATE','PPS_INDICATOR','PHARMACY_INDICATOR','TRANSPLANT_INDICATOR','RADIOLOGY_INDICATOR_','OUTPATIENT_SERVICES_INDICATOR','SURGERY_INDICATOR','DISCHARGE_DESTINATION','TYPE_OF_ADMISSION','INFRMTL_ENCTR_IND']
	coldict={'HMO_PAID_INDICATOR':HMO_PAID_INDICATOR,'INTENSIVE_CARE_INDICATOR':INTENSIVE_CARE_INDICATOR,'POA_DIAGNOSIS_INDICATOR_':POA_DIAGNOSIS_INDICATOR_,'CORONARY_CARE_INDICATOR':CORONARY_CARE_INDICATOR,'SPECIAL_UNIT_CHARACTER_CODE':SPECIAL_UNIT_CHARACTER_CODE,'SOURCE_OF_ADMISSION':SOURCE_OF_ADMISSION,'DRG_CODE':DRG_CODE,'INTERMEDIARY_NUMBER':INTERMEDIARY_NUMBER,'NCH_CLAIM_TYPE_CODE':NCH_CLAIM_TYPE_CODE,'AGE':AGE,'SEX':SEX,'RACE':RACE,'STATE':STATE,'PPS_INDICATOR':PPS_INDICATOR,'PHARMACY_INDICATOR':PHARMACY_INDICATOR,'TRANSPLANT_INDICATOR':TRANSPLANT_INDICATOR,'RADIOLOGY_INDICATOR_':RADIOLOGY_INDICATOR_,'OUTPATIENT_SERVICES_INDICATOR':OUTPATIENT_SERVICES_INDICATOR,'SURGERY_INDICATOR':SURGERY_INDICATOR,'DISCHARGE_DESTINATION':DISCHARGE_DESTINATION,'TYPE_OF_ADMISSION':TYPE_OF_ADMISSION,'INFRMTL_ENCTR_IND':INFRMTL_ENCTR_IND}
	colList2=['DIAGNOSIS_CODE_','PROCEDURE_CODE_','DIAGNOSIS_E_CODE_1','ADMITTING_DIAGNOSIS_CODE','POA_DIAGNOSIS_E_INDICATOR_1']
	colnameDict={}
	dicts=[]
	for i in range(len(colList1)):
		dicts.append({})
	for i in range(len(colList1)):
		itemlist=coldict[colList1[i]]
		for j in range(len(itemlist)):
			dicts[i][itemlist[j]]=j
		colnameDict[colList1[i]]=dicts[i]
	diagdict={}
	for i in range(len(diseases9)):
		diagdict[diseases9[i]]=i
	colnameDict['DIAGNOSIS_CODE_']=diagdict
	colnameDict['DIAGNOSIS_E_CODE_1']=diagdict
	colnameDict['ADMITTING_DIAGNOSIS_CODE']=diagdict
	colnameDict['POA_DIAGNOSIS_E_INDICATOR_1']=colnameDict['POA_DIAGNOSIS_INDICATOR_']
	procdict={}
	for i in range(len(procedures9)):
		procdict[procedures9[i]]=i
	colnameDict['PROCEDURE_CODE_']=procdict
	#special cases to clean up data
	colnameDict['HMO_PAID_INDICATOR']['0.0']=colnameDict['HMO_PAID_INDICATOR']['0']
	colnameDict['HMO_PAID_INDICATOR']['1.0']=colnameDict['HMO_PAID_INDICATOR']['1']
	colnameDict['HMO_PAID_INDICATOR']['2.0']=colnameDict['HMO_PAID_INDICATOR']['2']
	colnameDict['HMO_PAID_INDICATOR']['4.0']=colnameDict['HMO_PAID_INDICATOR']['4']
	for i in range(0,10):
		colnameDict['SOURCE_OF_ADMISSION'][str(i)+'.0']=colnameDict['SOURCE_OF_ADMISSION'][str(i)]
	colnameDict['POA_DIAGNOSIS_INDICATOR_']['0.0']=colnameDict['POA_DIAGNOSIS_INDICATOR_']['0']
	colnameDict['POA_DIAGNOSIS_INDICATOR_']['1.0']=colnameDict['POA_DIAGNOSIS_INDICATOR_']['1']

	return [colnameDict,colList1+colList2]

def selectVersion(inputpath, inputcsv, outputpath,outputcsv,version):
	import csv
	if(str(version)=='9' or str(version)=='9.0'):
		ver='0'
	if(str(version)=='0' or str(version)=='0.0'):
		ver='9'
	with open(inputpath+inputcsv) as f:
		filereader=csv.reader(f,delimiter=',')
		count=0
		with open(outputpath+outputcsv,'w') as g:
			gwriter=csv.writer(g,delimiter=',')
			for row in filereader:
				if(count==0):
					header=row
					gwriter.writerow(header)
					count=1
					cols=[]
					for i in header:
						if('VERSION' in i):
							cols.append(header.index(i))
				else:
					flag=0
					for i in cols:
						if(str(row[i])==ver or str(row[i])==ver+'0'):
							flag=1
							continue
					if(flag==0):
						gwriter.writerow(row)
	return

def separateCategoricalNumericalOutput(inputpath, inputcsv,configpath,configcsv,outputpath, outputcatcsv,outputnumcsv,outputycsv):
	import csv
	import pandas as pd
	import gc
	import numpy as np
	path1='/home/ubuntu/machlear/parul/steps/'
	filename='version9.csv'
	pathdest='/home/ubuntu/machlear/parul/steps/'
	inputcat=[]
	inputnum=[]
	y_vals=[]
	config=pd.read_csv(configpath+configcsv,engine='python')
	coldict={}
	cat=[]
	num=[]
	ycols=[]
	cols=[]
	for i in config['Col Name']:
		y=config.loc[(config['Col Name']==i)]['Use As (input(0), output(1), ignore(-1)']
		if(y.values[0]==0):
			x=config.loc[(config['Col Name']==i)]['Type (num (1) vs cat (0)']
			if(x.values[0]==0):
				coldict[i]='cat'
				cols.append(i)
			else:
				coldict[i]='num'
				cols.append(i)
		if(y.values[0]==1):
			coldict[i]='y'
			cols.append(i)
	df=pd.read_csv(inputpath+inputcsv,engine='python')
	header=list(df.columns.values)
	ignorecols=set(cols)-set(header)
	for i in cols:
		if(i in ignorecols):
			continue
		if(coldict[i]=='cat'):
			inputcat.append(df[i])
			cat.append(i)
		if(coldict[i]=='num'):
			inputnum.append(df[i])
			num.append(i)
		if(coldict[i]=='y'):
			y_vals.append(df[i])
			ycols.append(i)
	del df
	gc.collect()
	df=pd.DataFrame()
	input_cat=pd.DataFrame(np.transpose(np.array(inputcat)),columns=cat)
	inputcat=[]
	input_num=pd.DataFrame(np.transpose(np.array(inputnum)),columns=num)
	inputnum=[]
	yvals=pd.DataFrame(np.transpose(np.array(y_vals)),columns=ycols)
	y_vals=[]
	input_cat.to_csv(outputpath+outputcatcsv)
	input_num.to_csv(outputpath+outputnumcsv)
	yvals.to_csv(outputpath+outputycsv)
	return

def processNumInput(inputnumpath,inputnumcsv,outputnumpath,outputnumcsv):
	import csv
	import pandas as pd
	input_num=pd.read_csv(inputnumpath+inputnumcsv,engine='python')
	columns=list(input_num.columns.values)
	for i in columns:
		colmin=min(input_num[i])
		colmax=max(input_num[i])
		for j in range(input_num.shape[0]):
			if(colmax-colmin==0):
				if(colmax>=0 and colmax<=1):
					input_num[i][j]=colmax
				else:
					input_num[i][j]=1
			else:
				input_num[i][j]=(input_num[i][j]-colmin)/(colmax-colmin)
	input_num.to_csv(outputnumpath+outputnumcsv)
	return

def processCatInput(pathfile_dir,pathfile,inputcatpath, inputcatcsv,intermediate_files_dir,nlines,max_lines):
	import csv
	import pandas as pd
	import numpy as np
	import scipy.sparse
	from scipy.sparse import coo_matrix, csr_matrix,vstack,hstack
	import gc
	pathdict=getPaths(pathfile_dir,pathfile)
	[colnameDict,colnameList]=generateDict(pathdict['pathcodes'],pathdict['diseasecodesfile'],pathdict['proceduresfile'])
	colVecSize={}
	for i in colnameDict.keys():
		if(i=='INTENSIVE_CARE_INDICATOR' or i=='CORONARY_CARE_INDICATOR'):
			colVecSize[i]=len(colnameDict[i].values())+1
		elif(i=='RADIOLOGY_INDICATOR_'):
			colVecSize[i]=6
		elif(i=='SURGERY_INDICATOR' or i=='INFRMTL_ENCTR_IND'):
			colVecSize[i]=1
		else:
			colVecSize[i]=len(colnameDict[i].values())
	def getColVecSize(colname):
		return colVecSize[colname]
	def createOneHotVec(df,i,colname):
		vec=np.zeros(getColVecSize(colname))
		if(colname.startswith('DIAGNOSIS_CODE_')  or colname.startswith('POA_DIAGNOSIS_INDICATOR_')):
			for j in range(1,26):
				if(df[colname+str(j)][i] in colnameDict[colname].keys()):
					vec[colnameDict[colname][df[colname+str(j)][i]]]=1
		elif(colname.startswith('PROCEDURE_CODE_')):
			for j in range(1,26):
				y=df[colname+str(j)][i]
				x=str(y).split('.')[0]
				if(x in colnameDict[colname].keys()):
					vec[colnameDict[colname][x]]=1
		elif(colname=='INTENSIVE_CARE_INDICATOR' or colname=='CORONARY_CARE_INDICATOR'):
			if(not(df[colname][i] in colnameDict[colname].keys())):
				vec[-1]=1
			else:
				vec[colnameDict[colname][df[colname][i]]]=1
		elif(colname.startswith('RADIOLOGY_INDICATOR_')):
			for j in range(1,7):
				if(df[colname+str(j)][i] in colnameDict[colname].keys()):
					if(df[colname+str(j)][i]==1):
						vec[j-1]=1
		elif(colname=='SURGERY_INDICATOR'):
			if(df[colname][i] in colnameDict[colname].keys()):
				if(df[colname][i]==1):
					vec[0]=1
		elif(colname=='INFRMTL_ENCTR_IND'):
			if(df[colname][i] in colnameDict[colname].keys()):
				if(df[colname][i]=='Y'):
					vec[0]=1
		else:
			if(df[colname][i] in colnameDict[colname].keys()):
				vec[colnameDict[colname][df[colname][i]]]=1
		return vec
	def cleanOneHot(start_line):
		columnmatDict={}
		for i in colnameDict.keys():
			columnmatDict[i]=[]
#		nlines = int(sys.argv[2])
		df1=pd.read_csv(inputcatpath+inputcatcsv,nrows=2,engine='python')
		header=list(df1.columns.values)
		df=pd.read_csv(inputcatpath+inputcatcsv,skiprows=start_line,nrows=nlines,names=header,engine='python')
		del df1
		df.reset_index()
		for j in range(df.shape[0]):
			for i in colnameDict.keys():
				columnmatDict[i].append(createOneHotVec(df,j,i))
		del df
		gc.collect()
		df=pd.DataFrame()
		z=csr_matrix(columnmatDict['CORONARY_CARE_INDICATOR'])
		columnmatDict['CORONARY_CARE_INDICATOR']=[]
		for i in colnameList:
			if(i!='CORONARY_CARE_INDICATOR'):
				z=hstack([z,csr_matrix(columnmatDict[i])])
				columnmatDict[i]=[]
				gc.collect()
		savefile='OneHot_sparse_matrix_'+str(start_line)+'_'+str(nlines)+'.npz'
		scipy.sparse.save_npz(intermediate_files_dir+savefile,z)
		return
	start_line=0
	while (start_line<max_lines):
		cleanOneHot(start_line)
		start_line+=nlines
	return

def prepareInput(intermediate_files_dir,inputnumpath,processedinputnumcsv):
	import csv
	import pandas as pd
	import gc
	import numpy as np
	from sklearn import preprocessing
	from sklearn import linear_model
	from sklearn.metrics import r2_score
	import scipy.sparse
	from scipy.sparse import coo_matrix, csr_matrix,vstack,hstack
	import sys
	from os import listdir
	from os.path import isfile, join
	from sklearn.linear_model import ElasticNet
	from sklearn.cross_validation import KFold
	sparsematfilelist = [f for f in listdir(intermediate_files_dir) if (isfile(join(intermediate_files_dir, f)) and f.endswith('.npz'))]
	sparsematfilelist.sort()
	sparse_cat_train=scipy.sparse.load_npz(intermediate_files_dir+sparsematfilelist[0])
	for i in range(1,len(sparsematfilelist)-1):
		sparse_cat1=scipy.sparse.load_npz(intermediate_files_dir+sparsematfilelist[i])
		sparse_cat_train=vstack([sparse_cat_train,sparse_cat1])
	sparse_cat_test=scipy.sparse.load_npz(intermediate_files_dir+sparsematfilelist[-1])
	testlines=sparse_cat_test.shape[0]
	nlines=sparse_cat_train.shape[0]
	nummat_train=pd.read_csv(inputnumpath+processedinputnumcsv,nrows=nlines,engine='python')
	nummat_sparse_train=csr_matrix(nummat_train.values)
	numcols=list(nummat_train.columns.values)
	del nummat_train
	gc.collect()
	nummat_test=pd.read_csv(inputnumpath+processedinputnumcsv,skiprows=nlines,nrows=testlines,names=numcols,engine='python').values
	nummat_sparse_test=csr_matrix(nummat_test)
	del nummat_test
	gc.collect()
	inmat_train=hstack([sparse_cat_train,nummat_sparse_train])
	inmat_test=hstack([sparse_cat_test,nummat_sparse_test])
	sparse_cat_train=[]
	nummat_sparse_train=[]
	sparse_cat_test=[]
	nummat_sparse_test=[]
	return [inmat_train,inmat_test,nlines]




