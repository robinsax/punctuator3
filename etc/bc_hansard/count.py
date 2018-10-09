# coding: utf-8
'''
B.C. Hansard corpus magnitude check.
'''
import io
import os

def main():
	count = 0
	for filename in os.listdir('./data/bc_hansard'):
		with io.open('./data/bc_hansard/%s'%filename, encoding='utf-8') as text_file:
			count += len(text_file.read().split())
	print(count)

if __name__ == '__main__':
	main()
