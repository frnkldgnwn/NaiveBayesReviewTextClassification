import numpy as np
import pandas as pd
from Preprocessing import Preprocessing

class Weighting:
	def __init__(self, dataset):
		self.dataset = dataset
		pr = Preprocessing(self.dataset)
		self.stem, self.term = pr.preprocess()
	
	def set_bag_of_words(self):
		bag_of_words = []
		
		for row in self.term:
			for elem in row:
				if elem not in bag_of_words:
					bag_of_words.append(elem)
		
		self.print_arr("Bag of Words:", bag_of_words)
		
		self.bag_of_words = bag_of_words
	
	def get_bag_of_words(self):
		return self.bag_of_words
	
	def count_raw_tf(self):
		raw_tf = []
		
		for row in self.stem:
			row_raw_tf = []
			for elem in self.bag_of_words:
				row_raw_tf.append(row.count(elem))
			raw_tf.append(row_raw_tf)
		
		self.print_to_dataframe("Raw TF:", raw_tf)
		return raw_tf
	
	def count_logterm(self, raw_tf):
		logterm = [[1 + np.log10(raw_tf[i][j]) if raw_tf[i][j] != 0 else 0 for j in range(len(raw_tf[i]))] for i in range(len(raw_tf))]
		return logterm
	
	def count_df(self, raw_tf):
		df = []
		
		for i in range(len(self.bag_of_words)):
			df_row = 0
			for row in raw_tf:
				if row[i] != 0:
					df_row += 1
			df.append(df_row)
		
		return df
	
	def count_idf(self, raw_tf):
		df = self.count_df(raw_tf)
		idf = [np.log10(len(raw_tf) / elem) for elem in df]
		return idf
	
	def count_tfidf(self, raw_tf, idf):
		tfidf = raw_tf
		wtd = self.count_logterm(raw_tf)
		
		tfidf = [[wtd[i][j] * idf [j] for j in range(len(tfidf[i]))]for i in range(len(tfidf))]

		self.print_to_dataframe("TFIDF:", tfidf)
		return tfidf
	
	def print_arr(self, state, token):
		print(state)
		
		for i in range(len(token)):
			print("(", i, ")", token[i])
		
		print("\n")
	
	def print_to_dataframe(self, state, item):
		print(state)
		data = [[item[j][i] for j in range(len(item))] for i in range(len(item[0]))]
		data_frame = pd.DataFrame(data, index = self.bag_of_words)
		print(data_frame, "\n")

def export(item):
	data = [[item[j][i] for j in range(len(item))] for i in range(len(item[0]))]
	data_frame = pd.DataFrame(data, index = we.bag_of_words)
	
	export_csv = data_frame.to_csv (r'D:\Documents\KULIAH\Semester 5\TextMin\Tugas 2 Preprocessing\export_dataframe_tfidf.csv', index = we.bag_of_words, header = True, sep = ';', decimal = ',', float_format='%g')
