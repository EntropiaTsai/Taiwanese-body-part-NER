import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF  # 確保安裝 torchcrf

class ftmBERTfeatCRFModel(nn.Module):
    def __init__(self, model, num_labels, num_radicals, num_pos_tags, label2id,
                 radical_emb_dim=256, pos_emb_dim=256):
        super(ftmBERTfeatCRFModel, self).__init__()
        self.bert = model
        self.radical_embedding = nn.Embedding(num_radicals, radical_emb_dim)
        self.pos_embedding = nn.Embedding(num_pos_tags, pos_emb_dim)
        self.hidden2label = nn.Linear(768 + radical_emb_dim + pos_emb_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        # label mappings
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}

    def forward(self, input_ids, attention_mask, radical_features, pos_features):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embeddings = bert_outputs.last_hidden_state
        radical_embeddings = self.radical_embedding(radical_features)
        pos_embeddings = self.pos_embedding(pos_features)
        combined_embeddings = torch.cat((bert_embeddings, radical_embeddings, pos_embeddings), dim=-1)
        emissions = self.hidden2label(combined_embeddings)
        return emissions

    def loss(self, input_ids, attention_mask, labels, radical_features, pos_features):
        emissions = self.forward(input_ids, attention_mask, radical_features, pos_features)
        mask = attention_mask.bool()
        return -self.crf(emissions, labels, mask=mask, reduction='mean')

    def decode(self, input_ids, attention_mask, radical_features, pos_features):
        emissions = self.forward(input_ids, attention_mask, radical_features, pos_features)
        mask = attention_mask.bool()
        prediction = self.crf.decode(emissions, mask=mask)
        fixed_prediction = [self._fix_bio_labels(seq) for seq in prediction]
        return fixed_prediction

    def _fix_bio_labels(self, label_seq):
        fixed = []
        prev_type = None
        for label_id in label_seq:
            label = self.id2label[label_id]
            if label == 'O':
                fixed.append(label_id)
                prev_type = None
            elif label.startswith("B-"):
                fixed.append(label_id)
                prev_type = label[2:]
            elif label.startswith("I-"):
                current_type = label[2:]
                if prev_type == current_type:
                    fixed.append(label_id)
                else:
                    # 不合法的 I-*，改為 B-*
                    new_label = "B-" + current_type
                    fixed.append(self.label2id.get(new_label, label_id))
                    prev_type = current_type
            else:
                fixed.append(label_id)
                prev_type = None
        return fixed
