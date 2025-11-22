import MeCab

print("MeCab 테스트 시작")
tagger = MeCab.Tagger("")  # mecabrc에서 dicdir을 읽어 기본 사전 사용
text = "삼성전자가 서울에서 새로운 인공지능 센터를 열었다."
print(tagger.parse(text))
