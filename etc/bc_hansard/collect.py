# coding: utf-8
'''
A collection script for the British Columbia Hansard corpus (40th Parliament).

Transcript URLs are loaded from a data file as the index pages are dynamically
generated.
'''
import re
import io
import os
import sys
import json
import requests

from lxml import html

def parse(url):
	print(url)
	resp = requests.get(url)

	document = html.document_fromstring(resp.text)

	content = ''
	for speaker_tag in document.cssselect('.SpeakerBegins, .SpeakerContinues'):
		for spk in speaker_tag.cssselect('*:not(.Attribution)'):
			content += ' ' +  ' '.join(spk.itertext())

	#	Remove page numbers.
	content = re.sub(r'\[\s+Page\s[0-9]+\s+\]', ' ', content)
	#	Remove wack characters.
	for char in (u'”', u'“', u'…', '"'):
		content = content.replace(char, ' ')
	content = content.replace(u'’', "'")
	content = content.replace(u'—', '-')
	content = content.replace(u'–', '-')

	for x in ('Mr.', 'Ms.', 'B.C.'):
		content = content.replace(x, x[:-1])
	
	content = content.replace(',', ' ,COMMA ')
	content = content.replace(' -', ' -DASH ')
	content = content.replace(';', ' ;SEMICOLON ')
	content = content.replace(':', '  :COLON ')
	content = content.replace('. ', ' .PERIOD ')
	content = content.replace('?', ' ?QUESTIONMARK ')
	content = content.replace('!', ' !EXCLAIMATIONPOINT ')

	#	Normalize whitespace.
	content = re.sub(r'\s+', ' ', content).strip()
	#	Save result.
	with io.open('./data/bc_hansard/%s.txt'%resp.url.split('/')[-1][:10], 'w', encoding='utf-8') as f:
		f.write(content)

if not os.path.exists('./data/bc_hansard'):
	os.mkdir('./data/bc_hansard')

with open('./etc/bc_hansard/urls.json') as url_store:
	loaded_urls = json.load(url_store)
for url in loaded_urls:
	ps_find = list(re.finditer(r'\/(\w+)-parliament\/(\w+)-session', url))[0]
	parse('https://www.leg.bc.ca/content/Hansard/%s%s/%s.htm'%(
		ps_find.group(1), ps_find.group(2),
		url.split('/')[-1]
	))
