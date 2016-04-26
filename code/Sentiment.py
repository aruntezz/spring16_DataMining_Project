import re

from Neivebayes import *
types_of_categories =['Positive','Negative']
total_tweets=[]
stopwords={}

class Sentiment_analysis():	
	def __init__(self):
		self.feature_extractor = self.wordsplit
		
		

	def input_text(self,cl):
		sl = open("positive.txt","r").read()
		sl = sl.split("\n")
		for i in range(len(sl)):
			cl.train(sl[i],"Positive")
			total_tweets.append(sl[i])
		sl = open("neg.txt","r").read()
		sl = sl.split("\n")
		for i in range(len(sl)):
			cl.train(sl[i],"Negative")
			total_tweets.append(sl[i])
			
	def wordsplit(self,doc):
		splitter=re.compile('\\W*')
		words=[s.lower( ) for s in splitter.split(doc)]
		return dict([(w,1) for w in words])


	def addstopwords(self,cl):
		f3 =open("stopwords.txt","r").read()
		f3 = f3.split(",")
		for f in f3:
			stopwords[f]=1
			
	def start(self):
		nbcount=0

		cl1=Naivebayes(self.feature_extractor)
		self.addstopwords(cl1)
		self.input_text(cl1)
		f1=open("Result.html","w")
		nbcount2= self.final_Polariry(cl1,nbcount,f1)
		f1.write("<h3><center>So the polarity of the text is\t:\t"+nbcount2+"<br>")
		f1.write("Sentiment_analysis Has been done for test data</center></h3></body></html>")	
		f2=open("Result.html","r")
		#print '%d' % nbcount2;
	def final_Polariry(self,cl1,nbcount,f1):

		f = open("test.txt","r").read()
		f = f.split("\n")
		Pcount=0
		Ncount=0
		f1.write("<html><title>Sentiment Analysis</title><body bgcolor=\"pink\"><h1><center>Sentiment Analysis Using Machine learning Aproach</centre></color></h1>")
		for tweet in range(len(f)):
			result = cl1.classify(f[tweet],Pcount,Ncount)
			if(result=="Positive"):
				Pcount+=1
				f1.write(f[tweet]+"---"+result+"<br>")
			else:
				Ncount+=1
				f1.write(f[tweet]+"---"+result+"<br>")
				
		if(Pcount>Ncount):
			return "Positive"
		elif(Ncount>Pcount):
			return "Negative"
		else:
			return "Neutral"
	
		
def start_Sentiment_analysis():
	classifier = Sentiment_analysis()
	classifier.start()

if  __name__ =='__main__':start_Sentiment_analysis()
