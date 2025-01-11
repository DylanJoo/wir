from encoder import SparseEncoder
from transformers import AutoTokenizer
from transformers import AutoConfig

tokenizer = AutoTokenizer.from_pretrained('naver/splade-v3')

input1 = tokenizer('hello world', return_tensors='pt')
input2 = tokenizer('hello world', return_tensors='pt')
input3 = tokenizer('this is hello.', return_tensors='pt')

model = SparseEncoder.from_pretrained('naver/splade-v3',
    add_cross_attention=True,
    is_decoder=True,
    num_hidden_layers=12
)
output = model(**tokenizer('hello world', return_tensors='pt'))
print(output.reps.nonzero())
print(tokenizer.decode(output.reps.nonzero()[:, 1]))

output = model(**tokenizer('hello world', return_tensors='pt'),
    sub_input_ids=input3['input_ids'],
    sub_attention_mask=input3['attention_mask'],
    sub_token_type_ids=input3['token_type_ids']
)
print(output.reps.nonzero())
print(tokenizer.decode(output.reps.nonzero()[:, 1]))

model = SparseEncoder.from_pretrained('naver/splade-v3')
output = model(**tokenizer('hello world', return_tensors='pt'))
print(output.reps.nonzero())
print(tokenizer.decode(output.reps.nonzero()[:, 1]))
