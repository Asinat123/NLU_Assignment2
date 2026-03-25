import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd

############################
# LOAD & PREPROCESS DATA
############################

def load_data(file_path):
    with open(file_path, 'r') as f:
        names = [line.strip().lower() for line in f if line.strip()]
    
    chars = sorted(list(set(''.join(names))))
    
    char_to_int = {ch: i + 2 for i, ch in enumerate(chars)}
    char_to_int['<SOS>'] = 0
    char_to_int['<EOS>'] = 1
    
    int_to_char = {i: ch for ch, i in char_to_int.items()}
    
    return names, char_to_int, int_to_char

names, char_to_int, int_to_char = load_data("TrainingNames.txt")
VOCAB_SIZE = len(char_to_int)

############################
# HYPERPARAMETERS
############################

EMBED_DIM = 64
HIDDEN_SIZE = 128
LR = 0.002
EPOCHS = 10

############################
# MODEL 1: VANILLA RNN
############################

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

############################
# MODEL 2: BLSTM
############################

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

############################
# MODEL 3: RNN WITH ATTENTION
############################

class AttentionRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        
        attn_weights = torch.softmax(self.attn(out), dim=2)
        out = out * attn_weights
        
        out = self.fc(out)
        return out, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

############################
# PARAMETER REPORTING
############################

def report_model(model, name):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- {name} ---")
    print(f"Trainable Parameters: {params}")
    print(f"Hyperparameters: Hidden={HIDDEN_SIZE}, Embed={EMBED_DIM}, LR={LR}")

############################
# TRAINING FUNCTION
############################

def train_model(model, is_lstm=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for name in names:
            input_seq = [char_to_int['<SOS>']] + [char_to_int[c] for c in name]
            target_seq = [char_to_int[c] for c in name] + [char_to_int['<EOS>']]
            
            input_tensor = torch.tensor(input_seq).unsqueeze(0)
            target_tensor = torch.tensor(target_seq)
            
            optimizer.zero_grad()
            
            if is_lstm:
                output = model(input_tensor)
            else:
                hidden = model.init_hidden()
                output, hidden = model(input_tensor, hidden)
            
            loss = criterion(output.squeeze(0), target_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

############################
# INITIALIZE MODELS
############################

rnn_model = VanillaRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE)
lstm_model = BiLSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE)
attn_model = AttentionRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE)

############################
# REPORT MODELS
############################

report_model(rnn_model, "Vanilla RNN")
report_model(lstm_model, "BiLSTM")
report_model(attn_model, "Attention RNN")

############################
# TRAIN MODELS
############################

print("\nTraining Vanilla RNN...")
train_model(rnn_model)

print("\nTraining BiLSTM...")
train_model(lstm_model, is_lstm=True)

print("\nTraining Attention RNN...")
train_model(attn_model)


def generate_name(model, char_to_int, int_to_char, max_len=20, is_lstm=False):
    model.eval()
    with torch.no_grad():
        # Start with <SOS>
        input_seq = torch.tensor([char_to_int['<SOS>']]).unsqueeze(0)
        generated_name = ""
        
        # For RNNs, we need to manage the hidden state manually during generation
        hidden = None if is_lstm else model.init_hidden()
        
        for _ in range(max_len):
            if is_lstm:
                # Note: BiLSTM during inference usually uses only forward context
                output = model(input_seq)
            else:
                output, hidden = model(input_seq, hidden)
            
            # Get probabilities for the last character predicted
            probs = torch.softmax(output[0, -1], dim=0)
            char_idx = torch.multinomial(probs, 1).item()
            
            if char_idx == char_to_int['<EOS>']:
                break
                
            char = int_to_char[char_idx]
            generated_name += char
            
            # Use the predicted char as the next input
            input_seq = torch.tensor([char_idx]).unsqueeze(0)
            
        return generated_name

def evaluate_model(model, name_label, training_names, char_to_int, int_to_char, num_gen=100, is_lstm=False):
    generated_list = []
    for _ in range(num_gen):
        generated_list.append(generate_name(model, char_to_int, int_to_char, is_lstm=is_lstm))
    
    # 1. Diversity: Unique Generated / Total Generated
    unique_names = set(generated_list)
    diversity = len(unique_names) / num_gen
    
    # 2. Novelty: Generated names NOT in training set
    training_set = set(training_names)
    novel_names = [n for n in generated_list if n not in training_set]
    novelty_rate = len(novel_names) / num_gen
    
    print(f"\n--- Evaluation: {name_label} ---")
    print(f"Sample Names: {generated_list[:5]}")
    print(f"Diversity: {diversity:.2%}")
    print(f"Novelty Rate: {novelty_rate:.2%}")
    
    return generated_list

# Execute Evaluation
gen_rnn = evaluate_model(rnn_model, "Vanilla RNN", names, char_to_int, int_to_char)
gen_lstm = evaluate_model(lstm_model, "BiLSTM", names, char_to_int, int_to_char, is_lstm=True)
gen_attn = evaluate_model(attn_model, "Attention RNN", names, char_to_int, int_to_char)



def perform_qualitative_analysis(models_dict, training_names, char_to_int, int_to_char):
    print("### TASK 3: QUALITATIVE ANALYSIS DATA ###\n")
    
    analysis_data = []
    
    for name, model in models_dict.items():
        is_lstm = (name == "BiLSTM")
        generated = []
        
        # Generate 10 samples for analysis
        for _ in range(10):
            gen_name = generate_name(model, char_to_int, int_to_char, is_lstm=is_lstm)
            generated.append(gen_name if gen_name != "" else "[Empty String]")
        
        # Identify Failure Modes
        failures = []
        if is_lstm and any(g == "[Empty String]" or len(g) < 2 for g in generated):
            failures.append("Inference Mismatch (Output collapse)")
        if any(generated.count(x) > 1 for x in generated):
            failures.append("Repetitive Generation")
        if any(len(set(g)) < 3 and len(g) > 4 for g in generated):
            failures.append("Phonetic Looping")
            
        analysis_data.append({
            "Model": name,
            "Representative Samples": ", ".join(generated[:5]),
            "Failure Modes": " | ".join(failures) if failures else "None observed"
        })

    # Display as a clean Table
    df = pd.DataFrame(analysis_data)
    print(df.to_string(index=False))
    
    print("\n--- Discussion Points for Report ---")
    print("1. Realism: Compare 'kunya' (RNN) vs 'beeimhhi' (Attention).")
    print("2. BiLSTM Failure: Explain that BiLSTMs cannot 'see' the future during inference.")
    print("3. Attention: Note if the attention mechanism created overly complex character clusters.")

# Run the analysis
models_to_analyze = {
    "Vanilla RNN": rnn_model,
    "BiLSTM": lstm_model,
    "Attention RNN": attn_model
}

perform_qualitative_analysis(models_to_analyze, names, char_to_int, int_to_char)