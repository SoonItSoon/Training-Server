from ckonlpy.tag import Twitter
from konlpy.tag import Hannanum, Kkma, Komoran, Okt
from eunjeon import Mecab

test_text = "확진자와 접촉자는 다중이용시설 이용을 삼가하고, 사회적 거리두기 운동에 동참하며, 진료소와 마스크 착용을 자제해주시기 바랍니다."

# Customized Konlpy
twitter = Twitter()
twitter.add_dictionary(["확진자", "접촉자", "다중이용시설", "사회적", "거리두기", "진료소"], "Noun")
twitter.add_dictionary(["드립니다", "하시기", "해주시고", "해주시기", "지켜주십시오"], "Verb")
print(f"Customized Konlpy : {twitter.nouns(test_text)}")

# Hannanum
hannanum = Hannanum()
print(f"Hannanum : {hannanum.nouns(test_text)}")

# Kkma
kkma = Kkma()
print(f"Kkma : {kkma.nouns(test_text)}")

# Komoran
komoran = Komoran()
print(f"Komoran : {komoran.nouns(test_text)}")

# Okt
okt = Okt()
print(f"Okt : {okt.nouns(test_text)}")

# Mecab
mecab = Mecab()
print(f"Mecab : {mecab.nouns(test_text)}")

