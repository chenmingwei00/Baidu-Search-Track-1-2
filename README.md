# 【飞桨学习赛：百度搜索首届技术创新挑战赛：赛道一】第5名方案
> 子任务1涉及的答案抽取过程主要依赖答案片段与搜索query间语义相关性，却无法保证答案片段本身的正确性与可靠性。因此，在答案抽取之后需要设计答案验证方法，从抽取的多个答案片段中选择出大众认可度最高的高置信度答案进行最后的展示。给定一个搜索问题q和其对应的文档集合D，子任务2希望将所有文档基于其包含的答案观点一致性进行聚类，得到每个query下包含用户最公认答案的文档集合，保证深度智能问答系统最终答案的可信度。

## 1.项目描述
### 1.1 项目任务简要介绍：
```
给定用户query，给定query相关文档集合S,任务是从集合S中找到答案是正确的文档集合，所谓答案正确是现实生活中是客观对的，
例如：太阳系几大行星， 答案1：7 答案2：8； 答案2对
```
### 1.2 模型主要思路：
```
  a. 首先利用官方给定的答案两两组合语义是否一致的训练数据集训练一个语义相似度模型，输入结构为[CLS]query[SEP]answer1[SEP]answer2 ,利用[CLS]进行二分类
  b. 召回模型采用query+doc_text 作为输入，label为是否为正确答案训练二分类模型
  c. 预测结果，利用k-mean和相似度作为距离二分类或者多分类，去除类别个数少的，多的对应文档id作为正确答案
  d.然后利用召回模型对任务一中不包含答案的文档进行预测，标签为1的文档进行召回，最终得到提交文件，在第一次训练进行参数调节后，分数直接为0.786;
  之后就没有找到更好的方法继续提高
```
## 2. 项目结构

 与任务一目录结构类似，请参考:
   [任务一链接](https://github.com/chenmingwei00/Baidu-Search-Track-1)

## 3. 使用方式
 a. 使用方式可以直接运行本项目的AISTudio上的项目，生成新的副本，按照main.ipynb顺序指导运行生成提交文件
 b. 在AI Studio上[运行本项目](https://github.com/chenmingwei00/Baidu-Search-Track-1-2)  
