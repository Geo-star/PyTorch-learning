from torchtext import data
from torchtext.legacy.data import Field

# 定义文本切分方法，使用空格切分即可
mytokenize = lambda x: x.split()
# 定义将文本转化为张量的相关操作
TEXT = data.Field(sequential=True,  # 表明输入的文本是字符
				  tokensize=mytokenize,  # 使用自定义的分词方法
				  use_vocab=True,  # 创建一个词汇表
				  batch_first=True,  # batch优先的数据方式
				  fix_length=200  # 每个句子固定长度为200
				  )
