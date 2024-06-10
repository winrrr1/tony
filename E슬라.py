#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import IFrame

# PDF 파일을 불러오는 예제
IFrame("./E슬라-전기차-충전소-데이터-분석.pdf", width=1000, height=800)


# In[ ]:


from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import folium
import googlemaps
import json
import requests
from geopy.geocoders import Nominatim
import ggcodeClass
import pygwalker as pyg
from tqdm import tqdm
import time

if os.name == 'nt':
    font_family = "Malgun Gothic"
else:
    font_family = "AppleGothic"
    
sns.set(font=font_family, rc={"axes.unicode_minus" : False })

from IPython.display import set_matplotlib_formats


# In[ ]:


df = pd.read_excel('./한국환경공단_전기차 충전소 위치 및 운영정보(2024년 3월 기준).xlsx')

# 데이터프레임 확인
print(df.info())
print('-'*50)

# 설치년도별 충전소 개수 계산
ev_counts = df['설치년도'].value_counts().sort_index()
display(ev_counts) #설치년도별 개수  (설치년도별 차트)

# 막대 차트 생성
plt.figure(figsize=(10, 6)) #사이즈
ev_counts.plot(kind='line',rot=45, ) #설치년도별 차트 

# 차트 제목 및 축 레이블 설정
plt.title('설치년도별 충전소 차트')
plt.xlabel('설치년도')
plt.ylabel('충전소 개수', rotation=0, labelpad=40, ha='right')

# 이미지로 저장
plt.savefig("설치년도별 충전소 차트.jpg", format="jpg") 
plt.show()


# In[ ]:


df.columns


# In[ ]:


df['시도'][121:180]


# In[ ]:


# 주소에 '서울특별시'가 포함된 행만 선택
seoul_df = df[df['주소'].str.contains('서울특별시')]

print(seoul_df)


# In[ ]:


df[['기종(소)','기종(대)','충전소명','주소']]


# In[ ]:


# '설치년도'가 '2017'인 행을 필터링하고 '설치년도'와 '충전소명' 열만 선택
f_df = df[df['설치년도']==2017][['주소','충전소명']]
print(f_df)


# In[ ]:


df[['충전소명','주소']]


# In[ ]:


df.set_index(['설치년도','시도'])


# In[ ]:


df.groupby(['충전소명','주소']).sum()


# In[ ]:


df.set_index(['충전소명','주소'])


# In[ ]:


res=[]
for i in df['주소']:
    res.append(i)


# In[ ]:


'''
addr = []
lat = []
lng = []

max_items =  10 # 최대 처리할 항목 수 - !!
processed_items = 0

for i in res:
    addr.append(ggcodeF.get_addr(i))
    lat.append(ggcodeF.get_lat(i))
    lng.append(ggcodeF.get_lng(i))
    
    processed_items += 1
    if processed_items >= max_items:
        print(f"{max_items}개의 항목을 처리하여 중단합니다.")
        break
# 데이터프레임의 길이와 리스트의 길이를 일치시킵니다.
if len(addr) < len(df):
    addr.extend([None] * (len(df) - len(addr)))
if len(lat) < len(df):
    lat.extend([None] * (len(df) - len(lat)))
if len(lng) < len(df):
    lng.extend([None] * (len(df) - len(lng)))

df['주소'] = addr
df['위도'] = lat
df['경도'] = lng

print(addr[:5])  # 처음 5개의 주소 확인
print(lat[:5])   # 처음 5개의 위도 확인
print(lng[:5])   # 처음 5개의 경도 확인

# 데이터프레임의 처음 몇 개 행을 확인하려면 head() 메서드를 사용하세요.
print(df.head())
'''


# In[ ]:


map_df=df.groupby('시도').count()
map_df=map_df.reset_index()


# In[ ]:


geo_path='./Si_Do_map_utf8.json'
geo_data=json.load(open(geo_path, encoding= 'cp949'))


# In[ ]:


geo_data


# In[ ]:


data_cs=pd.read_excel('./한국환경공단_전기차 충전소 위치 및 운영정보(2024년 3월 기준).xlsx')


# In[ ]:


#data 부분만 

data_cs['시도'] = data_cs['시도'].replace('강원특별자치도', '강원도') #강원도 이름을 geodata에 맞게 변경
threshold_scale = [0, 1, 250, 1000, 2500, 10000, 20000, 30000] #범주 절댓값 지정

for i in range(2011,2023+1):
    #연도 순서대로 설정
    tmp_cs = data_cs[data_cs['설치년도'] == i]
    
    #각 시도별 설치 갯수
    tmp_csi = tmp_cs['시도'].value_counts().reset_index()
    tmp_csi.columns = ['시도', 'count']

    #폴리움 맵 설정
    map_cs = folium.Map(location=[36.3504567, 127.3848187], zoom_start=7)

    #Choropleth에 데이터 입력 후 맵에 넣기
    folium.Choropleth(
        geo_data=geo_data,  #경계 데이터 넣기
        name='choropleth',
        data=tmp_csi,
        columns=['시도', 'count'],
        key_on='feature.properties.CTP_KOR_NM', 
        fill_color='PuRd',
        fill_opacity=0.85,
        line_opacity=0.5,
        legend_name='설치 지점 수',
        threshold_scale=threshold_scale, #범주 절댓값 수동 지정
        nan_fill_color = 'white' #비거나 없는 값에 대한 디폴트 색상 흰색으로
    ).add_to(map_cs)
    
    map_cs.save(f'./전기차 충전소 연도별 지도_{i}.html') #결과물 지도를 html로 저장
    display(map_cs)  #결과물 지도를 출력


# 
# ## 빅카인즈 사이트 다운로드

# In[ ]:


import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as bs
import pandas as pd

options=Options()
options.add_experimental_option('detach',True) #화면이 꺼지지 않고 유지
options.add_argument('--start-maximized') # 화면을 최대화
service=Service(ChromeDriverManager().install()) #크롬드라이버 자동 설치

driver=webdriver.Chrome(service=service, options=options) #드라이버

url='https://www.bigkinds.or.kr/v2/news/search.do' #url 주소
driver.get(url)
time.sleep(3)

# 뉴스검색
driver.find_element(By.XPATH,'//*[@id="collapse-step-1"]').click() 

driver.find_element(By.XPATH,'//*[@id="total-search-key"]').click() 
time.sleep(1)
driver.find_element(By.XPATH,'//*[@id="total-search-key"]').send_keys('충전소 AND 보급률') #검색어 '충전소 보급률'
time.sleep(1)
driver.find_element(By.XPATH,'//*[@id="wrap"]/button').click() 
time.sleep(2)
driver.find_element(By.XPATH,'//*[@id="search-foot-div"]/div[2]/button[2]').click() 
time.sleep(2) 

# 3번 분석결과 및 시각화 
# driver.find_element(By.XPATH,'//*[@id="collapse-step-3"]').click()
# time.sleep(2)
# 엑셀 다운로드 로그인 아이디 비밀번호 필요함 
# driver.find_element(By.XPATH,'//*[@id="analytics-data-download"]/div[3]/button').click() #엑셀다운

news_data = []

for z in range(1, 11):  # Adjust the range as needed
    try:
        ev_for = f'//*[@id="news-results"]/div[{z}]/div/div[2]/a/div/strong/span'
        driver.find_element(By.XPATH, ev_for).click()
        time.sleep(5)

        #뷰티풀샵 
        page_source = driver.page_source
        soup = bs(page_source, 'html.parser')

        #본문
        ev_text = soup.select_one('div.news-view').text.strip()

        # 어펜드 
        news_data.append({
            'E슬라': ev_text
        })
        
        driver.find_element(By.XPATH,'//*[@id="news-detail-modal"]/div/div/button').click() #팝업x
        time.sleep(2)

    except Exception as e:
        print('다시')
        time.sleep(2)

#데이타 프레임으로 바꾸기 
cl_v = pd.DataFrame(news_data)

cl_v.to_csv('e슬라_news.csv', index=False, encoding='utf-8') #저장

driver.close()
driver.quit()
print(cl_v)


# ## 구글, 다음, 네이버 이미지 가져오기 (키워드=검색어 지정)

# In[ ]:


import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as bs
import pandas as pd
import urllib.request
import requests
import os

# Chrome 옵션 설정
options = Options()
options.add_experimental_option('detach', True)  # 브라우저 창을 닫아도 작업 유지
options.add_argument('--start-maximized')  # 브라우저 창을 최대화

# 검색할 키워드 설정
keyword = '인공지능 AND 테슬라'

# 네이버 이미지 검색 URL 설정
url1 = 'https://search.naver.com/search.naver?ssc=tab.image.all&where=image&sm=tab_jum&query='
url = url1 + keyword

# 네이버 이미지 수집
driver = webdriver.Chrome(service=service, options=options)  # Chrome 드라이버 초기화
driver.get(url)  # URL 열기
time.sleep(2)  # 페이지 로딩 대기
html = driver.page_source  # 현재 페이지 HTML 가져오기
soup = bs(html, 'html.parser')  # BeautifulSoup으로 HTML 파싱

# 이미지 태그 추출
imag_tags = soup.find_all('img', '_fe_image_tab_content_thumbnail_image')

res = []
# 이미지 주소 추출
for i in imag_tags:
    try:
        res.append(i['src'])
    except:
        res.append(i['data-lazysrc'])

# 저장할 폴더 생성
fold_name = f'{keyword}_네이버이미지'
os.makedirs(fold_name, exist_ok=True)

# 이미지 다운로드 및 저장
for idx, i in enumerate(res):
    urllib.request.urlretrieve(i, f'{fold_name}/{keyword}{idx:04d}.jpg')

# 브라우저 종료
driver.close()
driver.quit()

# 다음 이미지 검색 URL 설정
driver = webdriver.Chrome(service=service, options=options)
url = f'https://search.daum.net/search?w=img&nil_search=btn&DA=NTB&enc=utf8&q={keyword}'
driver.get(url)
time.sleep(2)
html = driver.page_source
soup = bs(html, 'html.parser')

res = []
# 다음 이미지 주소 추출
for i in soup.select('div.cont_image > div.wrap_thumb'):
    try:
        res.append(i.find('img')['src'])
    except:
        res.append(i.find('img')['data-original-src'])

# 저장할 폴더 생성
fold_name = f'{keyword}_다음이미지'
os.makedirs(fold_name, exist_ok=True)

# 이미지 다운로드 및 저장
for idx, i in enumerate(res):
    urllib.request.urlretrieve(i, f'{fold_name}/{keyword}{idx:04d}.jpg')

# 브라우저 종료
driver.close()
driver.quit()

# 구글 이미지 검색 URL 설정
driver = webdriver.Chrome(service=service, options=options)
url = f'https://www.google.com/search?tbm=isch&q={keyword}'
driver.get(url)
time.sleep(2)
html = driver.page_source
soup = bs(html, 'html.parser')

res=[]
# 구글 이미지 주소 추출
for i in soup.find_all('div','H8Rx8c'):
    res.append(i.find('img')['src'])

# 저장할 폴더 생성
fold_name = f'{keyword}_구글이미지'
os.makedirs(fold_name, exist_ok=True)

# 이미지 다운로드 및 저장
for idx, i in enumerate(res):
    urllib.request.urlretrieve(i, f'{fold_name}/{keyword}{idx:04d}.jpg')

# 브라우저 종료
driver.close()
driver.quit()


# ## 픽사베이 이미지 저장 

# In[ ]:


import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as bs
import pandas as pd
import urllib.request
import requests
import json
import os

#픽사베이  이미지 저장 

options=Options()
options.add_experimental_option('detach',True) #화면이 꺼지지 않고 유지
options.add_argument('--start-maximized') # 화면을 최대화
service=Service(ChromeDriverManager().install()) #크롬드라이버 자동 설치

driver=webdriver.Chrome(service=service, options=options)

def save_pixaImg(keyword, pages=1):
    for page in range(1, pages+1):
        url = f'https://pixabay.com/ko/photos/search/{keyword}/?pagi={page}'
        driver.get(url)

        time.sleep(1)
        html = driver.page_source
        soup = bs(html, 'html.parser')

        # 폴더 만들기
        fold_name = f'{keyword}_픽사베이'
        os.makedirs(fold_name, exist_ok=True)

        # 필요한 요소 가져오기
        img_tags = soup.find_all('div', class_='cell--B7yKd')

        for idx, i in enumerate(img_tags, start=1):
            text = json.loads(i.find('script').text) # JSON 파일을 load
            img_url = text['contentUrl'] # 이미지 주소 가져오기
            response = requests.get(img_url) # 이미지 요청
            file_name = f'{fold_name}/{keyword}{idx:04d}.jpg' # 저장할 파일 이름
            with open(file_name, 'wb') as f:
                f.write(response.content)

    driver.close()
    driver.quit()
    print('-----다운로드 성공-----')

if __name__ == '__main__':
    save_pixaImg('인공지능', pages=3)


# ## 네이버 블로그 저장

# In[ ]:


import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as bs
import pandas as pd
#네이버 블로그

options=Options()
options.add_experimental_option('detach',True) #화면이 꺼지지 않고 유지
options.add_argument('--start-maximized') # 화면을 최대화
service=Service(ChromeDriverManager().install()) #크롬드라이버 자동 설치

def get_naverBlog(search,pages=1):
    driver=webdriver.Chrome(service=service, options=options)

    res=[]
    for page in range(1,pages+1):
        url=f'https://section.blog.naver.com/Search/Post.naver?pageNo={page}&rangeType=ALL&orderBy=sim&keyword={search}'
        driver.get(url)
        time.sleep(2)
        elems=driver.find_elements(By.CLASS_NAME,'desc_inner')
        for elem in elems:
            tmp={}
            tmp['제목']=elem.text
            tmp['링크']=elem.get_attribute('href')
            res.append(tmp)
    df1=pd.DataFrame(res)
    driver.close()
    driver.quit()
    return df1
    
if __name__=='__main__':
    df1=get_naverBlog('인공지능',2)
    df1.to_csv('인공지능_네이버블로그.csv')


# In[ ]:


df.head()


# ## 여러가지 차트 그리기

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import Image
import seaborn as sns
import os

if os.name == 'nt':
    font_family = "Malgun Gothic"
else:
    font_family = "AppleGothic"
    
sns.set(font=font_family, rc={"axes.unicode_minus" : False })

from IPython.display import set_matplotlib_formats


# In[ ]:


tips = pd.read_csv('./환경부_전기자동차 급속충전기 보급 현황_20231231.csv', encoding='cp949')
tips


# In[ ]:


tips.columns


# In[ ]:


plt.figure(figsize=(10, 10))
sns.lmplot(x='년도', y='급속충전기 보급 수량',data=tips, height=5, aspect=2)
plt.title('')
# 이미지로 저장
plt.savefig("설치년도별 산점도.jpg", format="jpg")
plt.show()


# In[ ]:


sns.violinplot(x='년도', y='급속충전기 보급 수량', data=tips, height=5, aspect=2)
plt.title('')
plt.savefig("violin.jpg", format="jpg")
plt.show()


# In[ ]:


sns.countplot(tips['설명'])
plt.title('countplot')
plt.savefig("countplot.jpg", format="jpg")
plt.show()


# In[ ]:


df.columns


# In[ ]:


# 숫자형 데이터만 추출
numeric_data = tips.select_dtypes(include='number')

# 히트맵 생성
plt.title('heatmap')
sns.heatmap(numeric_data.corr(), annot=True, linewidths=1)
plt.savefig("heatmap.jpg", format="jpg")
plt.show()


# In[ ]:


x = ['Math', 'Programming', 'Data Science', 'Art', 'English', 'Physics']
y = [66, 80, 60, 50, 80, 10]

sns.barplot(x=x, y=y, alpha=0.8, hue=x, palette='YlGnBu', legend=False)

plt.ylabel('Scores')
plt.title('Subjects')
plt.show()


# In[ ]:


x = np.arange(0, 10, 0.1)
y_1 = 1 + np.sin(x)
y_2 = 1 + np.cos(x)

plt.plot(x, y_1, label='1+sin', color='blue', alpha=0.3)    #color : 컬러 옵션  #alpha :투명도 옵션
plt.plot(x, y_2, label='1+cos', color='red', alpha=0.7)

plt.xlabel('x value', fontsize=15)
plt.ylabel('y value', fontsize=15)
plt.title('sin and cos graph', fontsize=18)

plt.grid()
plt.legend()

plt.show()


# In[ ]:


N = 100000
bins = 30

x = np.random.randn(N)

fig, axs = plt.subplots(1, 3, 
                        sharey=True, 
                        tight_layout=True
                       )

fig.set_size_inches(12, 5)

axs[0].hist(x, bins=bins)
axs[1].hist(x, bins=bins*2)
axs[2].hist(x, bins=bins*4)

plt.show()


# In[ ]:


# project=3d로 설정합니다
ax = plt.axes(projection='3d')

sample_size = 100
x = np.cumsum(np.random.normal(0, 1, sample_size))
y = np.cumsum(np.random.normal(0, 1, sample_size))
z = np.cumsum(np.random.normal(0, 1, sample_size))

ax.plot3D(x, y, z, alpha=0.6, marker='o')

plt.title("ax.plot")
plt.show()


# In[ ]:


x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
x, y = np.meshgrid(x, y)

z = np.sin(np.sqrt(x**2 + y**2))

fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection='3d')

ax.contour3D(x, y, z, 20, cmap=plt.cm.rainbow)

plt.title("ax.contour3D")
plt.show()


# In[ ]:


x = [1,2,3,4,5,6,7,8,9]

sns.histplot(x, color="y", kde=True)  # kde=True는 커널 밀도 추정치를 함께 그리도록 설정합니다.
plt.show()


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df['시도'].value_counts().plot(kind='bar')
plt.savefig("시도.jpg", format="jpg")


# In[ ]:


df['군구'].value_counts().plot()
plt.savefig("군구.jpg", format="jpg")


# In[ ]:


df['주소'].describe()


# In[ ]:


display(df[['설치년도','주소','충전소명']])


# In[ ]:


df.iloc[0:500,0:6]


# ## 워드클라우드 

# In[ ]:


import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as bs
import pandas as pd
# CSV 파일 읽어오기
df2 = pd.read_csv('./e슬라_news.csv')
df2


# In[ ]:


# 전체 DataFrame의 모든 열을 문자열로 변환하기
df2 = df2.astype(str)

# 결과 확인
print(df2)


# In[ ]:


# 텍스트 파일 경로 설정
file_path = 'e슬라_news.txt'

# 텍스트 파일 읽기
with open(file_path, 'r', encoding='utf-8') as file:
    txt = file.read()

# 텍스트 파일 내용 출력
print(txt)


# In[ ]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
# 유니코드에서  음수 부호설정
plt.rc('axes', unicode_minus=False)


# In[ ]:


from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt   
# 단어 빈도수 계산
words = txt.split()
words = Counter(words)

# 딕셔너리로 변환
words = dict(words)

# 워드 클라우드 생성
wc=WordCloud(font_path="C:/windows/fonts/malgun.ttf",
             background_color='white',
             width=500,
             height=500,
             max_words=100000,
             max_font_size=200)
wc = wc.generate_from_frequencies(words)
# 워드 클라우드 이미지 표시
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='Lanczos') #Nearest-neighbor < Bilinear < Bicubic < Lanczos 선명도
plt.axis('off')
plt.savefig("단어.jpg", format="jpg")
plt.show()


# ## 형태소분석

# In[ ]:


get_ipython().system('pip install numpy')


# In[ ]:


from konlpy.tag import Okt,Kkma,Hannanum
import pandas as pd
import numpy as np


# In[ ]:


print(txt)


# In[ ]:


# 단어 빈도수 계산
words = txt.split()
words = Counter(words)

# 딕셔너리로 변환
words = dict(words)


# In[ ]:


#위의 워즈를 카운터 해준다. 
counter = Counter(words)
counter


# In[ ]:


from konlpy.tag import Okt,Kkma,Hannanum
okt=Okt()


# In[ ]:


okt.pos(txt)


# ## 형태소 단위 추출 

# In[ ]:


#텍스트를 형태소 단위로 나눈다. 
okt.morphs(txt)


# In[ ]:


#텍스트에서 명사만 추출
okt.nouns(txt)


# In[ ]:


#텍스트에서 어절만 추출
okt.phrases(txt)


# In[ ]:


txt


# In[ ]:


# 품사태깅(pos), stem=True 어간추출
pos=okt.pos(txt, stem=True)
pos


# In[ ]:


# 'Noun','Adjective','Verb' 빼기 
tmp_token=[word for word,p in pos if p in ['Noun','Adjective','Verb']]
tmp_token


# ###  stopword 지정

# In[ ]:


txt


# In[ ]:


# 명사 추출
okt=Okt()
n=okt.nouns(txt)
print(n)


# In[ ]:


df2.columns


# In[ ]:


text1=' '.join(df2['E슬라'])#제목만 추출
nouns1=okt.nouns(text1) # 제목에서 명사 추출
re_nouns1=' '.join(nouns1)
re_nouns1


# In[ ]:


# 제목에서 명사추출한 리스트 pickle로 저장하기(re1)
import pickle
with open('ev_news.pickle', 'wb') as f:
    pickle.dump(re_nouns1, f, pickle.HIGHEST_PROTOCOL)


# In[ ]:


# load
with open('ev_news.pickle', 'rb') as f:
    data0 = pickle.load(f)

data0


# In[ ]:


# stopwords 처리 전
wc = WordCloud("C:/windows/fonts/malgun.ttf",
               background_color = "white",
               max_words = 2000)
               #stopwords = stopwords)
wc = wc.generate(re_nouns1) # str자료만 가능, list자료는 안됨
plt.figure(figsize = (10, 10))
plt.imshow(wc) #interpolation = "bilinear")
plt.axis("off")
plt.savefig('./stopwords 처리전.jpg')
plt.show()


# In[ ]:


# 사람들이 자주 사용하는 모두의 스탑우드 - 불용어처리 
with open('./stopword_modu.txt','r') as f:
    stop_oneT=[i.strip() for i in list(f)]
print(stop_oneT)


# In[ ]:


#사용안하는 한글자 + 접미사 불용어 처리된 파일
m_re_nouns1=[i for i in n if i not in stop_oneT]
m_re_nouns1


# In[ ]:


# 카운터 처리해서 그래프에 넣는다. 
counter_m = Counter(m_re_nouns1)
counter_m


# - 상위 50개 단어 시각화

# In[ ]:


tmp_s=sorted(counter_m.items(), key=lambda x: x[1], reverse=True)
tmp_d=dict(tmp_s) #딕셔너리 처리 
top10=list(tmp_d)[:10] # 상위 10개만 보기 
top10


# In[ ]:


plt.figure(figsize=(15,8))
plt.bar(list(tmp_d.keys())[:50],list(tmp_d.values())[:50]) #딕셔너리 처리한 tmp_d
plt.xticks(rotation=90)
plt.savefig("50개단어.jpg", format="jpg")
plt.show()


# In[ ]:


tmp_d


# In[ ]:


df_top=pd.DataFrame(tmp_s).set_index(0)
df_top


# In[ ]:


# 상위 30개
df_top[:30].plot(kind='bar',figsize=(15,8))
plt.savefig('./ev_news상위50.jpg')
plt.show()


# In[ ]:


tmp_d.items()


# In[ ]:


X=list(tmp_d.keys())[:20]
y=list(tmp_d.values())[:20]
plt.barh(X,y)
plt.savefig('barh.jpg', format="jpg")


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 이미지 파일 읽기
img= plt.imread('./2.jpg')



# 이미지 배열 변환
mask_arr = np.array(img)

# 마스크 배열을 부호 없는 8비트 정수형으로 변환
mask_arr8 = (mask_arr * 255).astype(np.uint8)

# 워드 클라우드 생성
wc = WordCloud(font_path="C:/windows/fonts/malgun.ttf",
               background_color='white',
               mask=mask_arr8,  # 마스크 설정
               contour_width=3, contour_color='steelblue',
               width=500,
               height=500,
               max_words=100000,
               max_font_size=200)

wc = wc.generate_from_frequencies(tmp_d) # str자료만 가능, list자료는 안됨

# 워드 클라우드 생성 및 표시
plt.figure(figsize=(30, 10))
plt.imshow(wc, interpolation='Lanczos') #Nearest-neighbor < Bilinear < Bicubic < Lanczos
plt.axis('off')
plt.savefig("양자역학.jpg", format="jpg")
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 이미지 파일 읽기
img1 = plt.imread('./1.jpg')
img2 = plt.imread('./2.jpg')

# 워드 클라우드 생성 함수 정의
def c_wordcloud(img, frequencies):
    # 이미지 배열을 부호 없는 8비트 정수형으로 변환
    mask_arr = np.array(img)
    mask_arr8 = (mask_arr * 255).astype(np.uint8)
    
    # 워드 클라우드 생성
    wc = WordCloud(font_path="C:/windows/fonts/malgun.ttf",
                   background_color='white',
                   mask=mask_arr8,
                   contour_width=3, contour_color='steelblue',
                   width=500, height=500,
                   max_words=100000, max_font_size=200)
    wc = wc.generate_from_frequencies(frequencies)
    return wc

# 워드 클라우드 생성
wc1 = c_wordcloud(img1, tmp_d)
wc2 = c_wordcloud(img2, tmp_d)

# 서브플롯을 사용하여 두 개의 이미지를 하나의 figure에 표시
plt.figure(figsize=(30, 15))

# 첫 번째 이미지
plt.subplot(1, 2, 1)
plt.imshow(wc1, interpolation='Lanczos')
plt.title('과학자')
plt.axis('off')

# 두 번째 이미지
plt.subplot(1, 2, 2)
plt.imshow(wc2, interpolation='Lanczos')
plt.title('양자역학')
plt.axis('off')
plt.savefig("과학자+양자.jpg", format="jpg")
# 전체 출력
plt.show()


# In[ ]:


tmp_d


# In[ ]:


df


# In[ ]:


df2


# ### TF-IDF 순서대로 하세요. 정리 

# In[ ]:


import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as bs
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings
from tqdm import tqdm

# Suppress specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# Excel 파일 읽어오기
df3 = pd.read_excel('./ev빅카인즈100101231031.xlsx')
df3


# In[ ]:


df3.columns


# In[ ]:


ev_counts3 = df3['일자'].value_counts().sort_index()
ev_counts3


# In[ ]:


df3['본문']


# In[ ]:


# 1. 엔터(\n)으로 문장 분석
txt_res=[]
for v in df3['본문']:
    txt_res.extend(v.split('\n'))
tmpv=[v.strip() for v in txt_res]
tmpv
#2.  빈 문자열은 삭제
txt_res=[v for v in tmpv if len(v) !=0]
len(txt_res)
txt_res


# In[ ]:


from konlpy.tag import Okt,Kkma,Hannanum
import pandas as pd
import numpy as np
#3.  각 문서에서 명사추출
okt=Okt()
noun1=[okt.nouns(v) for v in txt_res]
noun1


# In[ ]:


#4. 한글자 불용어 처리
with open('./stopword_modu.txt','r') as f:
    stopw_mod=[v.strip() for v in list(f)]

last_noun1=[]
for v in noun1:
    last_noun1.append([w for w in v if w not in stopw_mod])
print(last_noun1)


# In[ ]:


# 5. 특정 단어 불용어 처리 (처리 안된 불용어 추가 )
stop_word1=['특','최','린수']
last_noun1=[]
for v in noun1:
    last_noun1.append([w for w in v if w not in stop_word1])
print(last_noun1)


# In[ ]:


#6. 전처리 완료
# last_nouns=[i for i in last_nouns if i]
last_noun1=[v for v in last_noun1 if len(v)!=0] #빈 리스트 삭제
last_noun1


# In[ ]:


#7. 모든 문서에 1번이상 등장한 단어
wordList=sorted(list(set([w for v in last_noun1 for w in v])))
wordList


# In[ ]:


len(wordList)


# In[ ]:


#8. TF 만들기 
dtm=[]
for noun in last_noun1:
    dtm.append([])
    for w in wordList:
        dtm[-1].append(noun.count(w))

tf_=pd.DataFrame(dtm,columns=wordList)
tf_


# In[ ]:


#9. IDF (전체 문서수/각단어가 등장한 문서수)
import numpy as np
# 전체 문서수
D=len(tf_)
print(D)
# 각 닥어가 등장한 문서수
df_t=tf_.astype('bool').sum()
df_t
idf=np.log(D/(df_t+1))
print(idf)


# In[ ]:


#10. TF_IDF (ef*idf)
tfidf=tf_*idf
tfidf


# In[ ]:


#저장
tfidf.to_csv('big_TFIDF.csv', index=False)


# In[ ]:


#######################################################################
### 방법1: numpy를 이용한 단어간 상관도 계산
#######################################################################
import numpy as np
import pandas as pd

word_corr=np.corrcoef(tf_,rowvar=False)
print(word_corr)

print('='*100)
vocab=tf_.columns
df4=pd.DataFrame(word_corr,columns=vocab)
df4.index=vocab
df4


# ## 네트워크 차트 

# In[ ]:


tf_


# In[ ]:


### 단어의 집합별 상관계수 값 리스트화 하기

words=tf_.columns
edges = []

for i in tqdm(range(len(words))):
    for j in range(i+1, len(words)):
        edges.append((words[i],words[j], word_corr[i,j]))


# In[ ]:


# 상관도 내림차순순으로 sort(reverse=True)
# 상관도 오름차순순으로 sort(reverse=False)
edges = sorted(edges, key=lambda x:x[2],reverse=True)
edges 


# In[ ]:


###########################################################
#상관계수 추출 : 상관게수를 네트워크 차트의 가중치로 사용함
#############################################################
weight_list=[x[2] for x in edges]   # weight_list = [weight for _, _, weight in edges] 도 사용가능
weight_list


# In[ ]:


##################################################
### 상관계수 시각화에 사용할 연결된 노드 이름 쌍
#################################################
edge_list = [(word1,word2) for word1, word2, weight in tqdm(edges)]
len(edge_list)


# In[ ]:


get_ipython().system('pip install networkx')


# ## 여기서부터 쉬프트 엔터치면 네트웍차트

# In[ ]:


import networkx as nx

G = nx.Graph()
rank=30

edge_set = set()  
for word1, word2, weight in edges[:rank]:   # 상관도가 높은 값에서 101개까지. 여기서는 15개만 있음으로 모두 나옴.
    G.add_edge(word1, word2, weight=weight) # 단어1,2 자료와 거기에 해당되는 상관도값(weight)
    edge_set.add((word1,word2))
len(edge_set)


# In[ ]:


# https://networkx.org/documentation/stable/tutorial.html
# https://95pbj.tistory.com/34   (한국어 사이트)


print('\n ▶ 노드갯수 출력')   # feature 갯수 (R, 분석, 시각화, 머신러닝, 파이썬, 차트)
print(G.number_of_nodes())

print('\n ▶노드값 출력')
print(G.nodes)

print('\n ▶ 엣지(word1,word2)값 출력')
print(G.edges)


## 디그리는 노드에서 분기하는 선의 갯수로서
# 여기서는 모든 노드와 연결되어 있음.(상관도를 모두 구함으로)
# 그래서 분석일때는 5개의 노드(n-1)개의 디그리가 생김.  분석->시각화, 분석->머신러닝, 분석->파이썬, 분석->차트, 분석-R
print('\n ▶ 디그리값 출력')  
print(G.degree)


print('\n ▶ 인접')
print(G.adj)
      
print('\n ▶ edges의 갯수')
print(G.number_of_edges())


print('\n ▶ 요약 ')
# print(nx.info(G))


# In[ ]:


#차트 그리기 전 필수 
tmp=np.sum(tf_)
print(tmp)
print('='*100)
nsize=tmp
nsize=300000*(nsize-min(nsize))/(max(nsize)-min(nsize))

print(nsize)


# ## 한글에서는  이걸로 오류 수정판

# In[ ]:


tf_


# In[ ]:


# 노드size: 단어빈도수(TF)
nsize


# In[ ]:


# 노드 degree 확인 : degree 키값으로 node그려짐..따라서...degree 키값 기반으로 size 재 조정함
G.degree


# In[ ]:


temp={k:nsize[k] for k in dict(G.degree)}
temp


# In[ ]:


nsize=[nsize[k] for k in dict(G.degree)]


# In[ ]:


nsize


# In[ ]:


###############################
### 상관계수 시각화
## 그래프를  그리기 위해서 준비해야 하는건
## 몇개의 노드를 준비할지,
# 그 노드에서 몇개의 디그리가 나올지
# 그리고 엣지(노드와 노드의 쌍)별 거리는 무엇으로 할지.
##############################

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.figure(figsize=(15,15)); plt.axis('off')

# fname = fm.FontProperties().get_name()


print(edges[:rank])
nx.draw_networkx(G, with_labels = True,
                 pos=nx.spring_layout(G,k=0.5),
                 font_family='Malgun Gothic',width=weight_list[:rank],edge_color='lightblue',font_size=15,
                 node_size=nsize)   # 상관관계값이 높을수록 가깝게 표시됨

# 이미지로 저장
plt.savefig("big_chart.jpg", format="jpg")
plt.show()


# ## 히트맵 오래걸림

# In[ ]:


tf_


# In[ ]:


# #################################################
# #### 히트맵을 이용한 상관도 출력
# #################################################
# # R과 머신러닝이 -1 의 상관도임
# # R이라는 글자가 나올때는 머신러닝이라는 글자가 전혀 나오지 않는다는 뜻
# import matplotlib.pyplot as plt
# # 한글 폰트 설정
# plt.rcParams['font.family'] = 'Malgun Gothic'
# # 유니코드에서  음수 부호설정
# plt.rc('axes', unicode_minus=False)
# import seaborn as sns
# plt.figure(figsize=(10,10))

# sns.heatmap(tf_[0:2].corr(),annot=True,cmap = 'PuBu')


# In[ ]:




