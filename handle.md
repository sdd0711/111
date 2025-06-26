
# 1. 多平台数据爬取模块
def scrape_data():
    """
    从多个平台爬取悬疑剧相关数据
    返回包含评论、评分和元数据的DataFrame
    """
    data = pd.DataFrame()
    
    # 示例：豆瓣爬取
    def scrape_douban(series_id):
        url = f"https://api.douban.com/v2/movie/subject/{series_id}/comments"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        comments = [item['content'] for item in response.json()['comments']]
        
        # 获取剧集信息
        detail_url = f"https://api.douban.com/v2/movie/subject/{series_id}"
        detail_res = requests.get(detail_url, headers=headers).json()
        
        return pd.DataFrame({
            'comment': comments,
            'rating': [detail_res['rating']['average']]*len(comments),
            'platform': ['douban']*len(comments),
            'title': [detail_res['title']]*len(comments)
        })
    
    # 添加更多平台爬取函数（如微博、B站等）
    # scrape_weibo(), scrape_bilibili()...
    
    # 爬取多部剧集
    for series_id in ['30174085', '30454281']:  # 示例剧集ID
        data = pd.concat([data, scrape_douban(series_id)])
    
    return data

# 2. 数据清洗与预处理
def clean_text(text):
    """文本清洗函数"""
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    # 移除连续空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(data):
    """数据预处理管道"""
    # 去重
    data = data.drop_duplicates(subset=['comment'])
    
    # 清洗文本
    data['cleaned_comment'] = data['comment'].apply(clean_text)
    
    # 中文分词和词性标注
    def tokenize(text):
        words = pseg.cut(text)
        return [word for word, flag in words if flag.startswith(('n', 'v', 'a'))]  # 保留名词、动词、形容词
    
    # 去停用词
    with open('stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])
    
    def remove_stopwords(words):
        return [word for word in words if word not in stopwords and len(word) > 1]
    
    data['tokens'] = data['cleaned_comment'].apply(tokenize).apply(remove_stopwords)
    
    return data

# 3. 文本特征提取
def extract_features(data):
    """从文本中提取多种特征"""
    # 情感分析
    def get_sentiment(text):
        # 使用TextBlob（需中英翻译）或使用中文情感词典
        # 这里使用简化版本
        positive_words = ['精彩', '好看', '烧脑', '悬疑', '演技']
        negative_words = ['烂', '无聊', '失望', '漏洞', '拖沓']
        score = sum([1 for word in text if word in positive_words]) - \
                sum([1 for word in text if word in negative_words])
        return score / len(text) if len(text) > 0 else 0
    
    data['sentiment'] = data['tokens'].apply(get_sentiment)
    
    # LDA主题模型
    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(data['tokens'].apply(' '.join))
    
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    topic_dist = lda.fit_transform(X)
    
    # 添加主题分布特征
    for i in range(5):
        data[f'topic_{i}'] = topic_dist[:, i]
    
    # 关键词提取（TF-IDF）
    tfidf = TfidfVectorizer(max_features=50)
    tfidf_matrix = tfidf.fit_transform(data['tokens'].apply(' '.join))
    keywords = tfidf.get_feature_names_out()
    
    # 添加关键词特征
    for word in keywords:
        data[f'kw_{word}'] = data['tokens'].apply(lambda x: 1 if word in x else 0)
    
    return data

# 4. 建模与分析
def build_model(data):
    """构建回归模型分析影响因素"""
    # 选择特征和目标变量
    features = data.filter(regex=r'^(sentiment|topic_|kw_)').columns.tolist()
    X = data[features]
    y = data['rating'].astype(float)
    
    # 多元线性回归
    model = LinearRegression()
    model.fit(X, y)
    
    # 交叉验证
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"模型R²分数: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
    
    # 输出特征重要性
    coef_df = pd.DataFrame({
        'feature': features,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\n特征影响程度排序:")
    print(coef_df.head(10))
    
    return model, coef_df

# 5. 结果可视化与报告生成
def generate_report(data, coef_df):
    """生成分析报告（简化版）"""
    # 情感分布可视化
    sentiment_dist = data['sentiment'].describe()
    
    # 主题关键词提取
    top_keywords = coef_df[coef_df['feature'].str.startswith('kw_')] \
        .sort_values('coefficient', ascending=False).head(5)
    
    # 生成建议
    positive_factors = coef_df[coef_df['coefficient'] > 0].head(3)['feature'].tolist()
    negative_factors = coef_df[coef_df['coefficient'] < 0].head(3)['feature'].tolist()
    
    report = f"""
    ======== 国产悬疑剧口碑分析报告 ========
    1. 情感分布:
        - 平均情感得分: {sentiment_dist['mean']:.2f}
        - 正面评价占比: {len(data[data['sentiment'] > 0.5])/len(data):.1%}
    
    2. 关键影响因素:
        - 正面因素: {', '.join(positive_factors)}
        - 负面因素: {', '.join(negative_factors)}
    
    3. 行业建议:
        - 创作: 加强{top_keywords.iloc[0]['feature'][3:]}相关剧情设计
        - 选角: 关注演员与{top_keywords.iloc[1]['feature'][3:]}特征的匹配度
        - 营销: 突出{positive_factors[0][3:]}和{positive_factors[1][3:]}的核心卖点
    """
    return report

# 主执行流程
if __name__ == "__main__":
    # 步骤1: 数据收集
    print("正在爬取数据...")
    raw_data = scrape_data()
    
    # 步骤2: 数据清洗
    print("数据清洗中...")
    cleaned_data = preprocess_data(raw_data)
    
    # 步骤3: 特征提取
    print("提取文本特征...")
    feature_data = extract_features(cleaned_data)
    
    # 步骤4: 建模分析
    print("构建分析模型...")
    model, coef_df = build_model(feature_data)
    
    # 步骤5: 生成报告
    report = generate_report(feature_data, coef_df)
    print(report)
    
    # 保存分析结果
    feature_data.to_csv('suspense_drama_analysis.csv', index=False)
    print("分析结果已保存!")    

序号	词语	频次
1	真的	294
2	剧情	285
3	演技	249
4	演员	247
5	故事	206
6	编剧	182
7	角色	172
8	警察	164
9	人物	163
10	原著	154
11	节奏	141
12	不错	140
13	悬疑	137
14	感觉	132
15	导演	131
16	国产	119
17	凶手	115
18	这部	113
19	江阳	110
20	镜头	109
21	观众	106
22	受害者	106
23	一部	103
24	朝阳	99
25	抄袭	98
26	孩子	98
27	逻辑	97
28	悬疑剧	97
29	好看	96
30	结局	96
31	改编	93
32	真相	92
33	喜欢	91
34	小说	88
35	大为	87
36	案件	86
37	剧本	85
38	杀人	84
39	表演	83
40	犯罪	82
41	迷雾	79
42	家庭	79
43	破案	77
44	真实	75
45	女性	73
46	情节	73
47	看完	70
48	电视剧	69
49	细节	69
50	一点	68
51	剧场	67
52	特别	65
53	结尾	65
54	叙事	64
55	很好	63
56	社会	63
57	一集	63
58	两个	63
59	希望	63
60	在线	63
61	秦昊	63
62	人性	62
63	隐秘	62
64	正义	61
65	刑侦	60
66	动机	60
67	值得	59
68	几个	59
69	三个	58
70	营销	58
71	作品	58
72	陈晓	58
73	推理	58
74	这剧	57
75	恶心	55
76	罪犯	55
77	顾己鸣	55
78	马伊	55
79	现实	54
80	看过	54
81	塑造	54
82	还原	54
83	白宇	54
84	五星	53
85	剪辑	52
86	全员	52
87	小孩	52
88	角落	52
89	理解	51
90	整体	51
91	张东升	51
92	时间	50
93	实在	50
94	过程	48
95	黑暗	48
96	期待	47
97	确实	46
98	反派	45
99	剧里	45
100	出轨	45
101	线索	45
102	生活	45
103	严良	45
104	转场	44
105	质感	44
106	老师	44
107	那种	44
108	长夜难明	44
109	名字	43
110	刑警	43
111	地方	43
112	电影	43
113	两集	42
114	发现	42
115	真真	41
116	案子	41
117	配乐	41
118	赵今麦	40
119	第一集	40
120	几集	40
121	性格	40
122	杀人犯	39
123	台词	39
124	只能	38
125	美化	38
126	江娜	38
127	父亲	37
128	男主	37
129	刻画	37
130	沉默	37
131	东西	35
132	题材	35
133	手法	35
134	永远	35
135	情感	35
136	陈建斌	35
137	松本	35
138	清张	35
139	主角	34
140	一种	34
141	肉联厂	34
142	本来	34
143	感情	34
144	精彩	34
145	光明	34
146	心理	33
147	关系	33
148	环境	33
149	中国	32
150	烂尾	32
151	老公	32
152	老卫	32
153	团队	31
154	好好	31
155	可惜	31
156	降智	31
157	内容	31
158	视角	31
159	制作	31
160	十三年	31
161	女人	30
162	王千源	30
163	水平	30
164	强行	30
165	选择	30
166	过审	30
167	方式	30
168	不行	30
169	世界	30
170	好像	30
171	终于	30
172	女主	29
173	剧组	29
174	主创	29
175	坏人	29
176	完美	28
177	原型	28
178	有人	28
179	变态	28
180	交代	28
181	嫌疑人	28
182	场景	28
183	紧凑	28
184	失望	28
185	之间	27
186	爱奇艺	27
187	那段	27
188	拖沓	27
189	全剧	27
190	表现	27
191	人生	27
192	蓝盈莹	27
193	垃圾	26
194	受害人	26
195	好人	26
196	设计	26
197	剧集	26
198	一场	26
199	影响	26
200	太好了	26
201	到位	26
202	设定	26
203	一句	26
204	无证	26
205	算是	25
206	呈现	25
207	漂白	25
208	努力	25
209	剧中	25
210	拍摄	25
211	显得	25
212	开头	25
213	审查	25
214	语言	25
215	峥嵘	25
216	怀孕	25
217	美剧	24
218	洗白	24
219	本剧	24
220	水准	24
221	为啥	24
222	片子	24
223	越来越	24
224	宁理	24
225	小演员	24
226	遗憾	24
227	原因	23
228	明白	23
229	男的	23
230	人血馒头	23
231	成功	23
232	看着	23
233	三星	23
234	剧作	23
235	基因	23
236	网剧	23
237	花椒	23
238	少年	23
239	普普	23
240	紫金	23
241	智商	22
242	刻意	22
243	肯定	22
244	年代	22
245	最终	22
246	儿子	22
247	惊喜	22
248	没什么	22
249	晓芙	22
250	反转	22
251	张颂文	22
252	三集	21
253	身上	21
254	一群	21
255	注水	21
256	记者	21
257	第一次	21
258	大人	21
259	印象	21
260	想要	21
261	原生	21
262	氛围	21
263	表达	21
264	评分	21
265	演得	21
266	童年	21
267	感动	21
268	最佳	21
269	童话	21
270	坏小孩	21
271	廖凡	21
272	感受	20
273	宣传	20
274	双线	20
275	案情	20
276	事件	20
277	猎奇	20
278	影视	20
279	不到	20
280	时代	20
281	狂飙	20
282	原作	20
283	没想到	20
284	铺垫	20
285	目标	20
286	很棒	20
287	完整	20
288	爱情	20
289	观感	20
290	郭京飞	19
291	风格	19
292	报警	19
293	尺度	19
294	主线	19
295	可怜	19
296	漫长	19
297	每次	19
298	出彩	19
299	推进	19
300	一口气	19
301	杨哲	19
302	烟雾弹	19
303	优秀	19
304	成长	19
305	作家	19
306	六页	19
307	无语	18
308	戏份	18
309	冲突	18
310	老婆	18
311	所有人	18
312	悬念	18
313	刻板	18
314	形象	18
315	不好	18
316	导致	18
317	调查	18
318	丈夫	18
319	莫名其妙	18
320	更是	18
321	价值	18
322	无聊	18
323	男人	18
324	第四集	18
325	突兀	18
326	程度	18
327	主题	18
328	眼神	18

