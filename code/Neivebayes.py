from Sentiment import *

class Naivebayes():
	def __init__(self,getfeatures,filename=None):
		self.featurecount={}
		self.categorycount={}
		self.getfeatures=getfeatures

	def feature_count(self,f,cat):
		if f in self.featurecount and cat in self.featurecount[f]:
			return float(self.featurecount[f][cat])
		return 0.0
		
	def catcount(self,cat):
		if cat in self.categorycount:
			return float(self.categorycount[cat])
		return 0
		
	def train(self,item,cat):
		features=self.getfeatures(item)
		for f in features:
			if f in stopwords:
				continue
			else:
				self.featurecount.setdefault(f,{})
				self.featurecount[f].setdefault(cat,0)
				self.featurecount[f][cat]+=1
				self.categorycount.setdefault(cat,0)
				self.categorycount[cat]+=1
				
	def fprob(self,f,cat):
		if self.catcount(cat)==0: return 0
		return self.feature_count(f,cat)/self.catcount(cat)
		
	def cprob(self,f,cat):
		clf=self.fprob(f,cat)
		if clf==0: return 0
		freqsum=sum([self.fprob(f,c) for c in self.categorycount.keys()])	# total frequency of feature
		p=clf/(freqsum)	
		return p
		
	def Naive_prob(self,item,cat):	#calculates  probability
		p=1
		features=self.getfeatures(item)
		for f in features:
			basicprob=self.cprob(f,cat)
			totals=sum([self.feature_count(f,c) for c in self.categorycount.keys()])
			p *= ((0.5)+(totals*basicprob))/(1.0+totals)
		return p	
			
	def classify(self,item,scount,pcount,default='None'):
		best=default
		p1 = self.Naive_prob(item,"Positive")
		p2 = self.Naive_prob(item,"Negative")

		if(p1>p2):
			best="Positive"
			
		elif(p2>p1):
			best="Negative"
		else:
			best="Neutral"
		return best
