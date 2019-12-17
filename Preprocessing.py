import numpy as np
import pandas as pd
import re
import csv
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class Preprocessing:
	#ini konstruktor untuk menset dataset awal yang akan di preprocess
	def __init__(self, dataset):
		self.dataset = dataset
	
	#ini method untuk melakukan parsing dataset untuk dipisah menjadi kategori dan teks reviewnya
	def parsing(self):
		kategori = [row['Kategori'] for index, row in self.dataset.iterrows()]
		review = [row['Review'] for index, row in self.dataset.iterrows()]
		self.print_arr("Setelah Parsing:", review)
		return kategori, review
	
	#ini method untuk memisahkan teks menjadi token, yang berisi pemisahan kata, pembersihan angka dan simbol, dan case folding menjadi huruf kecil
	def tokenisasi(self, review):
		token = [re.split('\W+', elem) for elem in review]
		self.print_arr("Setelah Tokenisasi:", token)

		token = self.hapus_angka_simbol(token)
		self.print_arr("Setelah Angka dan Simbol Dihapus:", token)
		
		case_fold = self.case_folding(token)
		self.print_arr("Setelah Case Folding:", case_fold)
		
		return case_fold

	#ini method untuk menghapus angka dan simbol
	def hapus_angka_simbol(self, token):
		[[row.remove(elem) for elem in row if any(c.isdigit() for c in elem)] for row in token]
		return token

	#ini method untuk mengubah huruf menjadi huruf kecil (non-kapital)
	def case_folding(self, token):
		case_fold = [[w.lower() for w in row] for row in token]
		return case_fold

	#ini method untuk menghapus elemen-elemen tidak penting
	def cleaning(self, token):
		[[row.remove(elem) for elem in row if elem == ''] for row in token]
		return token

	#ini method untuk melakukan penghapusan kata-kata yang ada di stoplist (menggunakan stoplist tala)
	def filtering(self, token):
		with open("stopword_list_tala.csv", newline = "") as csvfile:
			stoplist = list(csv.reader(csvfile))

		stoplist = np.asarray(stoplist)
		removed = [[t for t in row if t not in stoplist] for row in token]
		self.print_arr("Setelah Filtering:", removed)
		return removed

	#ini method untuk melakukan stemming kata menjadi kata dasarnya dengan menggunakan bantuan library Sastrawi
	def stemming(self, token):
		stemmer = StemmerFactory().create_stemmer()
		stemmed = [stemmer.stem(' '.join(row)).split() for row in token]
		self.print_arr("Setelah Stemming:", stemmed)
		return stemmed
	
	#ini method untuk menghapus kata-kata yang ganda
	def terming(self, token):
		term = []
			
		for row in token:
			row_term = []
			for elem in row:
				if elem not in row_term:
					row_term.append(elem)
			term.append(row_term)
		
		self.print_arr("Setelah Term:", term)
		return term

	#ini method untuk melakukan print terhadap array sehingga mudah dilihat
	def print_arr(self, state, token):
		print(state)
		[print("(", i, ")", token[i]) for i in range(len(token))]
		print("\n")
	
	#ini method untuk melakukan preprocessing yang memanggil semua method di atas
	def preprocess(self):
		print("Data Frame Awal:\n", self.dataset, "\n")
		
		kategori, review = self.parsing()
		token = self.tokenisasi(review)
		token = self.cleaning(token)
		token = self.filtering(token)
		token = self.stemming(token)
		term = self.terming(token)
		
		data = {'Kategori': kategori, 'Term (Review)': term}
		data_frame = pd.DataFrame(data)
		
		print("Data Frame Setelah Preprocessing:\n", data_frame, "\n")
		return token, term
