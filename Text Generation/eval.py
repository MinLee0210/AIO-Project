from train import *
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# PERPLEXITY
def compute_perplexity(logits, targets):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(targets, logits)
    perplexity = np.exp(np.mean(loss))
    return perplexity

# BLEU-score:
def compute_bleu(predicted, targets):
    predicted_strings = []
    for seq in predicted:
        seq = np.argmax(seq, axis=1)
        string_seq = ' '.join([tokenizer.sequences_to_texts([[token]])[0] for token in seq if token != 0])
        predicted_strings.append(string_seq)

    target_strings = []
    for seq in targets:
        seq = seq.numpy().tolist()
        string_seq = ' '.join([tokenizer.sequences_to_texts([[token]])[0] for token in seq if token != 0])
        target_strings.append(string_seq)

    blue_score = corpus_bleu(target_strings, predicted_strings)
    return blue_score

def plot_result():
    train_loss, train_acc = history.history['loss'], history.history['masked_accuracy']
    val_loss, val_acc = history.history['val_loss'], history.history['val_masked_accuracy']

    plt.figure(figsize=(10, 10))

    plt.subplot (2, 2, 1)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.plot(train_loss, color='green')

    plt.subplot (2, 2, 2)
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.plot(train_acc, color='orange')

    plt.subplot (2, 2, 3)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.plot(val_loss, color='green')

    plt.subplot (2, 2, 4)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.plot(val_acc, color='orange')

    plt.show()