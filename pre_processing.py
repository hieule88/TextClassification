# Load the model in fairseq
from fairseq.models.roberta import RobertaModel
import glob2
from tqdm import tqdm
import utils

dns_home = '/storage/hieuld/NLP'
phoBERT = RobertaModel.from_pretrained('PhoBERT_base_fairseq', checkpoint_file='model.pt')
phoBERT.eval()  # disable dropout (or leave in train mode to finetune

train_path = 'Data/Train_Full/*/*.txt'
test_path = 'Data/Test_Full/*/*.txt'


text_train, label_train = utils.make_data(train_path)
text_test, label_test = utils.make_data(test_path)

# Lưu lại các files

utils._save_pkl('Data/text_train.pkl', text_train)
utils._save_pkl('Data/label_train.pkl', label_train)
utils._save_pkl('Data/text_test.pkl', text_test)
utils._save_pkl('Data/label_test.pkl', label_test)



