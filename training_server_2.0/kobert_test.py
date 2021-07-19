import torch
from torch import nn
from torch.utils.data import Dataset
from kobert.utils import get_tokenizer
import numpy as np
import pandas as pd
import gluonnlp as nlp
from kobert.pytorch_kobert import get_pytorch_kobert_model
from tqdm import tqdm


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=10,  # 분류 개수에 따라 수정
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc


def testModel():
    torch.multiprocessing.freeze_support()
    torch.cuda.empty_cache()

    bertmodel, vocab = get_pytorch_kobert_model()
    # 기본 Bert Tokenizer 사용
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    max_len = 64

    device = None
    model = BERTClassifier(bertmodel, dr_rate=0.5)
    # GPU 사용 시
    if torch.cuda.is_available():
        print("GPU 사용...")
        device = torch.device("cuda")
        model.load_state_dict(torch.load("model/kobert_model-classification_state-dict8.pt"), strict=False)
        model.to(device)
    # CPU 사용 시
    else:
        print("CPU 사용...")
        device = torch.device("cpu")
        model.load_state_dict(torch.load("model/kobert_model-classification_state-dict.pt", map_location=device))
    # model = torch.load("model_covid-classification.pt")
    model.eval()

    # 테스트 문장 예측
    # mapping_dict = {0: '보건소방문', 1: '안내', 2: '확진자발생'}
    # tttt = [["[중대본]4.2.~9. 성남시 분당구 소재 도우미 이용 노래방 방문자·근무자는 가까운 보건소 선별진료소에서 코로나19 검사를 받으시기 바랍니다(☎120,1339)", 0],
    #         ["[용인시청] 4월10일 확진자18명[(용인 2243~2260번) ▶처인구2, 기흥구3, 수지구12, 수원시1] 발생하였습니다. corona.yongin.go.kr", 2],
    #         ["[순천시청] 코로나19 감염이 인근(목포, 광주)에서 지속 발생하고 있습니다. 개개인이 방역주체가 되어 마스크 착용 등 방역수칙을 반드시 준수 바랍니다. ", 1]]
    # unseen_test = pd.DataFrame([[test_sentence, test_label]], columns=[['MESSAGE', 'CATEGORY']])
    # unseen_test = pd.DataFrame(tttt, columns=[['MESSAGE', 'CATEGORY']])
    # unseen_values = unseen_test.values
    unseen_test = nlp.data.TSVDataset("txt/alertMsg_test_top10.txt", field_indices=[1, 2], num_discard_samples=1)
    test_set = BERTDataset(unseen_test, 0, 1, tok, max_len, True, False)
    test_input = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=5)

    con_matrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    accuracy = 0
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_input)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)

        max_vals, max_indices = torch.max(out, 1)
        accuracy += calc_accuracy(out, label)
        con_matrix[label][max_indices.item()] += 1
        # print(f"\n{tttt[batch_id][0]} : {mapping_dict[max_indices.item()]}")

    accuracy /= (batch_id + 1)
    precision = 0
    recall = 0
    f1_score = 0
    for i in range(0, 10):
        print(con_matrix[i])
        TP = con_matrix[i][i]
        FP = 0
        FN = 0
        for j in range(0, 10):
            if j != i:
                FP += con_matrix[j][i]
                FN += con_matrix[i][j]
        precision += (TP / (TP + FP))
        recall += (TP / (TP + FN))
    precision /= 10
    recall /= 10
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"Accuracy : {accuracy:.6f}\nPrecision : {precision:.6f}\nRecall : {recall:.6f}\nF1-Score : {f1_score:.6f}")


if __name__ == '__main__':
    testModel()
