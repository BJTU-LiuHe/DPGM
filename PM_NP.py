import numpy as np

def norm_c(Qs,num_matches,num_node1,num_node2,num_pair):
	ret=np.zeros_like(Qs)
	for idx in range(num_pair):
		QCur=Qs[idx][:num_matches[idx]]
		QCur=QCur.reshape(num_node1[idx],num_node2[idx])
		Q_normC=np.expand_dims(np.sum(QCur,axis=0),axis=0)
		retCur=QCur/Q_normC
		ret[idx,:num_matches[idx],:]=retCur.reshape(-1,1)
	return ret

def norm_r(Qs,num_matches,num_node1,num_node2,num_pair):
	ret = np.zeros_like(Qs)
	for idx in range(num_pair):
		QCur = Qs[idx][:num_matches[idx]]
		QCur = QCur.reshape(num_node1[idx], num_node2[idx])
		Q_normC = np.expand_dims(np.sum(QCur, axis=1),axis=1)
		retCur = QCur / Q_normC
		ret[idx, :num_matches[idx],:] = retCur.reshape(-1,1)
	return ret
iteration=100
num_pair=3
num_node1=[5,3,4]
num_node2=[5,4,3]
num_matches=[25,12,12]
Ks=[np.random.randint(0,10,(25,25)),np.random.randint(0,10,(12,12)),np.random.randint(0,10,(12,12))]

max_match=max([x.shape[0] for x in Ks])

K_tensor=np.zeros((num_pair,max_match,max_match))

for idx in range(num_pair):
	K_tensor[idx,:num_matches[idx],:num_matches[idx]]=Ks[idx]

Ps=np.zeros((num_pair,max_match,1))
for idx in range(num_pair):
	Ps[idx,:num_matches[idx],:]+=1

for ite in range(iteration):
	Ps_last=Ps
	Qs=np.matmul(K_tensor,Ps)

	if ite % 2==1:
		Ps=norm_c(Qs,num_matches,num_node1,num_node2,num_pair)
	else:
		Ps=norm_r(Qs,num_matches,num_node1,num_node2,num_pair)  #[num_pair,max_match]

	K_tensor=K_tensor*((Ps/(Ps_last+10e-6))*np.ones((num_pair,max_match,1)))

print(Ps)


