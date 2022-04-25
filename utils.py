import json
import torch
from tqdm import tqdm
from sklearn import metrics


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def open_config_file(filepath):
    with open(filepath) as jsonfile:
        pdict = json.load(jsonfile)
        params = AttrDict(pdict)
    return params


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_metrics(probs, labels, threshold: float = 0.5):
    probs, labels = probs.numpy(), labels.numpy()
    preds = probs > threshold
    acc = metrics.accuracy_score(labels, preds)
    auc = metrics.roc_auc_score(labels, probs)
    precision = metrics.precision_score(labels, preds, zero_division=0)
    recall = metrics.recall_score(labels, preds)
    
    metrics_dict = {'acc': acc, 'auc': auc, 'precision': precision, 'recall': recall}
    return metrics_dict


def run_training(epoch, model, train_dataset, optimizer, criterion, params, threshold=0.5):

    model.train()
    epoch_loss = 0
    probs = []
    labels = []

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=False
    )

    with tqdm(train_loader,
              desc=(f'Train - Epoch {epoch}'),
              unit=' tensor',
              ncols=80,
              unit_scale=params.batch_size) as t:

        for i, batch in enumerate(t):

            optimizer.zero_grad()
            _, sparse_tensor, label = batch
            sparse_tensor, label = sparse_tensor.cuda(), label.cuda()
            logits = model(sparse_tensor)
            loss = criterion(logits, label.float())
            loss.backward()
            optimizer.step()

            prob = torch.sigmoid(logits)
            probs.extend(prob[:,0].clone().tolist())
            labels.extend(label.clone().tolist())

            epoch_loss += loss.item()
        
        metrics = get_metrics(probs, labels, threshold)
        avg_loss = epoch_loss / len(train_loader)
        
        return avg_loss, metrics


def run_validation(epoch, model, val_dataset, criterion, params, threshold=0.5):

    model.eval()
    epoch_loss = 0
    probs = []
    labels = []

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        shuffle=False
    )

    with tqdm(val_loader,
              desc=(f'Validation - Epoch {epoch}'),
              unit=' tensor',
              ncols=80,
              unit_scale=params.batch_size) as t:

        with torch.no_grad():
            
            for i, batch in enumerate(t):

                _, sparse_tensor, label = batch
                sparse_tensor, label = sparse_tensor.cuda(), label.cuda()
                logits = model(sparse_tensor)
                loss = criterion(logits, label.float())
                
                prob = torch.sigmoid(logits)
                probs.extend(prob[:,0].clone().tolist())
                labels.extend(label.clone().tolist())

                epoch_loss += loss.item()
        
        metrics = get_metrics(probs, labels, threshold)
        avg_loss = epoch_loss / len(val_loader)
        
        return avg_loss, metrics


def run_test(model, test_dataset, params, threshold=0.5):

    model.eval()
    instance_indices = []
    probs = []
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.test_batch_size, shuffle=False)    

    with tqdm(test_loader,
             desc=(f'Test '),
             unit=' tensor',
             ncols=80,
             unit_scale=params.test_batch_size) as t:

        with torch.no_grad():

            for i, batch in enumerate(t):

                index, image, lymph_count, label = batch
                image, lymph_count, label = image.cuda(), lymph_count.cuda(), label.cuda()
                logits = model(image)                
                prob = torch.sigmoid(logits)
                probs.extend(prob[:,0].clone().tolist())
                instance_indices.extend(list(index))
                
    test_dataset.df.loc[instance_indices, 'prob'] = probs

    preds_dict = {'Id': list(patient_ids), 'Predicted': list(preds.numpy())}
    test_predictions_df = pd.DataFrame.from_dict(test_preds_dict)
    
    return test_predictions_df