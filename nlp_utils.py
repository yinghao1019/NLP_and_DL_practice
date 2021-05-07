from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import random
import torch
import tqdm
import os


class Model_pipeline:
    def __init__(self, model, train_iter, optimizer, scheduler, val_iter, model_configs):
        self.model = model
        self.model_configs = model_configs
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.optimizer = optimizer
        self.scheduler = scheduler

    def amp_training(self, epochs, save_model_dir, inp_name, trg_name,
                     max_norm=None, teach_ratio=None, eval_func=None, per_ep_eval=5):

        epoch_pgb = tqdm.trange(epochs, desc='EPOCHS')
        scaler = GradScaler()
        # check save model dir
        if not os.path.isdir(save_model_dir):
            print('Model dir is not existed')
            os.makedirs(save_model_dir)
        else:
            print('Model dir already existed')
        # build storage loss array
        ep_train_loss = []
        ep_val_loss = []

        # Training model
        self.model.zero_grad()
        for ep in epoch_pgb:
            # build iter pipe
            iter_loss = 0

            self.model.train()
            for batch in self.train_iter:

                # select model input
                inputs = {inp_name: vars(batch)[inp_name],
                          trg_name: vars(batch)[trg_name]}

                with autocast():
                    if teach_ratio:
                        output, loss = self.model(**inputs,
                                                  teach_ratio=teach_ratio)
                    else:
                        output, loss = self.model(**inputs)

                iter_loss += loss.item()

                # avoid grad underflow
                scaler.scale(loss).backward()

                # avoid grad expolding
                scaler.unscale_(self.optimizer)
                if max_norm:
                    clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

                # update weight & lr
                scaler.step(self.optimizer)
                scaler.update()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()  # clear grad

            print(f'[{ep+1}/{epochs}] training loss:\
                   {iter_loss/len(self.train_iter)}')
            ep_train_loss.append(iter_loss/len(self.train_iter))

            # evaluate model
            if (per_ep_eval > 0) and ((ep+1) % per_ep_eval == 0):
                metrics = self.evaluate_model(
                    self.val_iter, eval_func, inp_name, trg_name)
                print(f'Eval metrics for trained model in {ep+1}')
                for k, v in metrics.items():
                    print(f'{k}:{v}')
                ep_val_loss.append(metrics['eval_loss'])

            # saving model checkpoint
            self.save_modelCkp(save_model_dir, self.model,
                               self.optimizer, self.model_configs, epochs)
        epoch_pgb.close()
        return ep_train_loss, ep_val_loss  # train loss

    def evaluate_model(self, data_iter, eval_func, inp_name, trg_name):
        eval_loss = 0
        val_pgb = tqdm.tqdm(data_iter)
        total_metrics = defaultdict(int)

        self.model.eval()
        for batch in val_pgb:
            with torch.no_grad():
                # select model input
                inputs = {inp_name: vars(batch)[inp_name],
                          trg_name: vars(batch)[trg_name]}

                outputs, loss = self.model(**inputs)

            eval_loss += loss.item()
            # eval
            batch_metrics = eval_func(outputs, vars(batch))
            # update metrics
            for n, v in batch_metrics.items():
                total_metrics[n] += v
        # scale for epoch level
        total_metrics['eval_loss'] = eval_loss
        for k, v in total_metrics.items():
            total_metrics[k] = v/len(self.val_iter)
        return total_metrics

    def save_modelCkp(self, model_dir, model, optimizer, configs, epochs):
        # save model state and config
        print(f'Start save Model to dir:{model_dir}')
        model_state = {'model': model.state_dict(
        ), 'optimizer': optimizer.state_dict(), 'epochs': epochs}

        torch.save(model_state, os.path.join(model_dir, 'model_state.pt'))
        torch.save(configs, os.path.join(model_dir, 'model_conifg.pt'))
        print('Save Model success!')

# set random seed


def set_rnd_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.deterministic = True


# set weight except bias value for N(0,1) value and bias weight for 0
def init_model_weights(m):
    for named, params in m.named_parameters():
        if 'weight' in named:
            torch.nn.init.normal_(params.data, 0, 0.01)
        else:
            torch.nn.init.constant_(params.data, 0)


def count_parameters(model):
    """It's will count total params num for model."""
    return sum(p.numel()for p in model.parameters() if p.requires_grad)


# load Model
def load_model(model_dir, model_path, device):
    if not os.path.isdir(model_dir):
        raise Exception('Model dir is not existed')
    else:
        load_path = os.path.join(model_dir, model_path)
        try:
            state_dict = torch.load(load_path, map_location=device)
            print('Read model file is successed')
            return state_dict
        except:
            raise Exception('Model file is loss...')


def get_classMetrics(outputs, data):
    assert 'label' in data
    # convert to class idx
    binary_c = True if outputs.size()[1] == 1 else False
    eval_type = 'binary' if binary_c else 'weighted'
    # get correct label
    if binary_c:
        preds = torch.where(torch.sigmoid(outputs) >= 0.5, 1, 0)
        preds = preds.squeeze(dim=1).cpu().detach().numpy()
    else:
        preds = torch.argmax(torch.ouputs).cpu().detach().numpy()

    labels = data['label'].cpu().detach().numpy()  # get correct label

    acc = accuracy_score(labels, preds)
    f_score = f1_score(labels, preds, average=eval_type)
    precision = precision_score(labels, preds, average=eval_type)
    recall = recall_score(labels, preds, average=eval_type)

    return {
        'acc': acc, 'precision': precision,
        'f1_score': f_score, 'recall': recall,
    }


def get_class_inference(model, input_sents, nlp_pipe, min_length, Binary):
    """
    This function is use to predict input_sents sentiment by
    using trained model.

    Parameters:
    model(torch.Module):The model that you want to train.Note that if you want to 
                        use GPU for training,you should put it in GPU before pass to
                        this arg.
    input_sents(str or list of token):You want to predict sentences.Note that the str should be Unicode.
    Field(torchtext.field):This Field object will convert sent's tokens to numerical. 
    min_length(int):set sentence minimize length.If your len(input_sents) lower than min_length.
                   It will pad sents length as same as min_length. 
    Return:
     Positive or Negative sentiment(int)
    """
    if isinstance(input_sents, str):
        input_tokens = nlp_pipe.preprocess(input_sents)
    else:
        input_tokens = [t.lower() for t in input_sents if t.isalpha()]

    # pad_tokens
    if len(input_tokens) < min_length:
        input_tokens += ['<pad>']*(min_length-len(input_tokens))
    # convert to tensor
    input_tensor = nlp_pipe.numericalize([input_tokens])

    if isinstance(model, nn.Module):
        if next(model.parameters()).is_cuda:
            input_tensor = input_tensor.cuda()

        model.eval()
        with torch.no_grad():
            output, _ = model(text=input_tensor)

        # convert to predict idx
        if Binary:
            predict = torch.where(torch.sigmoid(output) >= 0.5, 1, 0)
            predict = predict.squeeze(dim=1).cpu().detach().numpy()
        else:
            predict = torch.argmax(torch.ouput).cpu().detach().numpy()

    else:
        if next(model.DL_model.parameters()).is_cuda:
            input_tensor = input_tensor.cuda()
            predict = model.predict(input_tensor)

    return predict
