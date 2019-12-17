import pandas as pd
import numpy as np
from Weighting import Weighting
from Preprocessing import Preprocessing

class Classification:
	def __init__(self):
		self.prior = []
		self.likelihood = []
		self.feature = []
	
	def train(self, dataset):
		self.class_trained = [row['Kategori'] for index, row in dataset.iterrows()]
		we = Weighting(dataset)
		
		we.set_bag_of_words()
		self.feature = we.get_bag_of_words()
		
		raw_tf = we.count_raw_tf();
		we.print_to_dataframe("Raw Term Frequency:", raw_tf)
		
		sum_term_in_documents = pd.DataFrame([{'DOKUMEN' : index + 1, 'SUM' : sum(raw_tf[index])} for index, row in dataset.iterrows()])
		print(sum_term_in_documents)
		
		count_class = pd.DataFrame([{'CLASS' : self.class_trained[index], 'SUM' : row['SUM']} for index, row in sum_term_in_documents.iterrows()]).groupby('CLASS').sum().reset_index()
		print(count_class)
		
		sum_term_plus_class = [{'CLASS' : self.class_trained[index], 'RAW_TF' : list(map(int, raw_tf[index]))} for index, row in dataset.iterrows()]
		df = pd.DataFrame(sum_term_plus_class)
		count_term_class = pd.DataFrame(df.pop('RAW_TF').values.tolist())
		count_term_class.columns = self.feature
		count_term_class.insert(0, "CLASS", [elem['CLASS'] for elem in sum_term_plus_class], True)
		count_term_class = count_term_class.groupby('CLASS', as_index = False).sum()
		print(count_term_class)
		
		temp_count_class = count_class.drop('CLASS', 1)
		temp_count_term_class = count_term_class.drop('CLASS', 1)
		likelihood = [[(items + 1) / (temp_count_class.loc[index, 'SUM'] + len(self.feature)) for names, items in row.iteritems()] for index, row in temp_count_term_class.iterrows()]
		likelihood = pd.DataFrame(likelihood)
		likelihood.columns = self.feature
		likelihood.insert(0, "CLASS", [row['CLASS'] for index, row in count_term_class.iterrows()], True)
		self.likelihood = likelihood
		print(likelihood)
		
		sum_buruk = 0
		sum_medium = 0
		sum_bagus = 0
		sum_all = dataset['Kategori'].count()
		
		for index, row in dataset.iterrows():
			if row['Kategori'] == 'Buruk':
				sum_buruk += 1
			if row['Kategori'] == 'Medium':
				sum_medium += 1
			if row['Kategori'] == 'Bagus':
				sum_bagus += 1
		
		prior = [sum_buruk / sum_all, sum_medium / sum_all, sum_bagus / sum_all]
		prior = pd.DataFrame(prior)
		prior.columns = ['Prior']
		prior.insert(0, "CLASS", [row['CLASS'] for index, row in count_term_class.iterrows()], True)
		self.prior = prior
		print(prior)
		
	def test(self, dataset):
		pr = Preprocessing(dataset)
		stem, term = pr.preprocess()
		df_train = pd.DataFrame({'Dokumen': ['D ' + str(i) for i in range(dataset['Kategori'].count())], 'Stem': stem})
		print(df_train,  "\n")
		
		class_result = []
		
		for index, document in df_train.iterrows():
			posteriors = [row['Prior'] for index, row in self.prior.iterrows()]
			likelihoods = [np.prod([likelihood for likelihood in row[[elem for elem in document['Stem']]]]) for index, row in self.likelihood.iterrows()]
			posteriors = [posteriors[i] * likelihoods[i] for i in range(len(posteriors))]
			
			if posteriors[0] == max(posteriors):
				class_result.append('Bagus')
			elif posteriors[1] == max(posteriors):
				class_result.append('Buruk')
			elif posteriors[2] == max(posteriors):
				class_result.append('Medium')
			
		print("Hasil Klasifikasi Test: \n", pd.DataFrame({'Dokumen': ['D ' + str(i) for i in range(dataset['Kategori'].count())], 'CLASS': class_result}), "\n")
		return class_result
	
	def count_accuracy(self, predicted, real):
		counter = [1  if predicted[i] == real[i] else 0 for i in range(len(predicted))]
		return (sum(counter)/len(counter)) * 100

if __name__ == '__main__':
	cl = Classification()
	nama_kolom = ['Kategori', 'Review']
	dataset_train = pd.read_csv('DataSetReviewRembangan.csv', delimiter = ';', names = nama_kolom)
	
	cl.train(dataset_train)
	
	nama_kolom = ['Kategori', 'Review']
	dataset_test = pd.read_csv('DataTest.csv', delimiter = ';', names = nama_kolom)
	
	class_result = cl.test(dataset_test)
	
	print("Akurasi Klasifikasi Testing: ", 
		str(cl.count_accuracy(class_result, [row['Kategori'] for index, row in dataset_test.iterrows()])) + '%')
