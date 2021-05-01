import utils
from tqdm import tqdm
import torch
from fairseq.data import Dictionary
import numpy as np
from sklearn.preprocessing import LabelEncoder

# test , train dạng văn bản
text_train = utils._load_pkl('Data/text_train.pkl')
label_train = utils._load_pkl('Data/label_train.pkl')
text_test = utils._load_pkl('Data/text_test.pkl')
label_test = utils._load_pkl('Data/label_test.pkl')

max_sequence_length = 256
# Load the dictionary  
vocab = Dictionary()
vocab.add_from_file("PhoBERT_base_transformers/dict.txt")

def convert_lines(lines, vocab, bpe):
  ''' 
  lines: list các văn bản input 
  vocab: từ điển dùng để encoding subwords
  bpe: 
  '''
  # Khởi tạo ma trận output
  outputs = np.zeros((len(lines), max_sequence_length), dtype=np.int32) # --> shape (number_lines, max_seq_len)
  # Index của các token cls (đầu câu), eos (cuối câu), padding (padding token)
  cls_id = 0
  eos_id = 2
  pad_id = 1


  for idx, row in tqdm(enumerate(lines), total=len(lines)): 
    # Mã hóa subwords theo byte pair encoding(bpe)
    subwords = bpe.encode('<s> '+ row +' </s>')
    input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
    # Truncate input nếu độ dài vượt quá max_seq_len
    if len(input_ids) > max_sequence_length: 
      input_ids = input_ids[:max_sequence_length] 
      input_ids[-1] = eos_id
    else:
      # Padding nếu độ dài câu chưa bằng max_seq_len
      input_ids = input_ids + [pad_id, ]*(max_sequence_length - len(input_ids))
    
    outputs[idx,:] = np.array(input_ids)
  return outputs

# Test encode lines
lines = ['Học_sinh được nghỉ học bắt dầu từ tháng 3 để tránh dịch covid-19', 'số lượng ca nhiễm bệnh đã giảm bắt đầu từ tháng 5 nhờ biện pháp mạnh tay']
[x1, x2] = convert_lines(lines, vocab, phoBERT.bpe)
print('x1 tensor encode: {}, shape: {}'.format(x1[:10], x1.size))

print(vocab[1])

print('x1 tensor decode: ', phoBERT.decode(torch.tensor(x1))[:103])

X = convert_lines(text_train, vocab, phoBERT.bpe)

lb = LabelEncoder()
lb.fit(label_train)
y = lb.fit_transform(label_train)
print(lb.classes_)
print('Top 5 classes indices: ', y[:5])

# Save dữ liệu
_save_pkl('PhoBERT_pretrain/X1.pkl', X)
_save_pkl('PhoBERT_pretrain/y1.pkl', y)
_save_pkl('PhoBERT_pretrain/labelEncoder1.pkl', lb)

# Load lại dữ liệu
X = _load_pkl('PhoBERT_pretrain/X1.pkl')
y = _load_pkl('PhoBERT_pretrain/y1.pkl')

print('length of X: ', len(X))
print('length of y: ', len(y))

