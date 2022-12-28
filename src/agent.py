import os
from os.path import join
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from datetime import datetime
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from src import tool, logger
from src import dataset as DataSet
from src import net as Model
from src import action as Action

class Trainer:

    def __init__(self, args):
        # cuda setting
        os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']

        # dir setting
        self.model_dir = args['model_dir']
        self.best_model_dir = args['best_model_dir']
        tool.check_mkdir(self.model_dir)
        tool.check_mkdir(self.best_model_dir)

        # dataset setting
        self.dataloader = DataSet.get_dataloader(args)
        self.no_eval = args['no_eval']   #no need to eval ，default：false
        self.personal_eval = args['personal_eval']  # true ， default：Patient-level
        self.img_size = args['img_size']
        self.batch_size=args['batch_size']
        self.is_Train = False
        self.i = args['desc'].split('_')[-1] # index of ensembled model
        args['mean'] = self.dataloader.mean
        args['std'] = self.dataloader.std
        args['num_classes'] = self.dataloader.num_classes   # num_classes =2

        # basic setting
        self.opt_type = args['optimizer']
        self.lr = args['lr']
        self.lr_epoch = args['lr_epoch']
        self.epoch = args['epoch']
        self.weight = args['weight']
        self.eval_best = 0
        self.eval_best_recall = 0
        self.eval_best_auc = 0
        self.eval_best_epoch = 0
        self.save_cm = args['save_cm']  # save confusion matrix ,default ：false

        self.timeStamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        self.model_desc = '{}_{}_{}_{}'. \
            format(args['dataset'], args['model'], args['action'], args['desc'])
        self.tb_desc = '{}_{}_{}'. \
            format(args['model'],args['desc'],self.timeStamp)
        self.model_pkl = self.model_desc + '.ckpt'

        # logger setup
        self.pblog = logger.get_pblog()
        self.pblog.total = self.epoch
        self.tblog = SummaryWriter(join(args['tb_dir'], self.tb_desc))   # TensorBoard

        # model setup
        self.action = Action.get_action(args)
        self.model = Model.get_net(args)   # model=se_resnet_152

        # load best model
        if args['pre_train']:
            state_dir = join(self.best_model_dir,'Best_'+self.model_desc)
            state = torch.load(state_dir, map_location='cpu')
            self.model.load_state_dict(state['net'])


        self.model.cuda()
        # self.action.save_graph(self.model, self.img_size, self.tblog,
        #                        self.pblog)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            # ism: IS using Multiple gpus
            self.ism = True
        else:
            self.ism = False

    def __del__(self):
        if hasattr(self, 'tb_log'):
            self.tblog.close()


    def train(self):
        self.pblog.info(self.model_desc)
        optimizer = None
        self.is_Train = True
        for epoch in range(self.epoch):
            # get optimizer
            temp = self.action.update_opt(epoch, self.model, self.opt_type,
                                          self.lr, self.lr_epoch)
            if temp is not None:
                optimizer = temp

            self.model.train()
            loss_l = []
            loss_n = []
            dl_len = len(self.dataloader.train)
            ll = len(self.action.eval_legend)
            c_right = np.zeros(ll, np.float32)
            c_sum = np.zeros(ll, np.float32)
            main_loss = 0
            for idx, item in enumerate(self.dataloader.train):
                tx, ty = item['image'], item['label']
                tx, ty = tx.cuda(non_blocking=True), ty.cuda(non_blocking=True)
                # get network output logits
                logits = self.action.cal_logits(tx, self.model)
                # cal loss : return loss,
                loss = self.action.cal_loss(ty, logits, self.weight)
                # cal acc
                right_e, sum_e = self.action.cal_eval(ty, logits)
                # backward
                optimizer.zero_grad()
                loss[0].backward()
                optimizer.step()

                c_right += right_e
                c_sum += sum_e
                loss_l.append([ii.item() for ii in loss])
                loss_n.append(ty.size(0))
                main_loss += loss[0].item()

                self.pblog.pb(idx, dl_len, 'Loss: %.5f | Acc: %.3f%%' % (
                    main_loss / (idx + 1), c_right / c_sum))

            loss_l = np.array(loss_l).T
            loss_n = np.array(loss_n)
            loss = (loss_l * loss_n).sum(axis=1) / loss_n.sum()
            c_res = c_right / c_sum

            msg = 'Epoch: {:>3}'.format(epoch)
            loss_scalars = self.action.cal_scalars(loss,
                                                   self.action.loss_legend, msg,
                                                   self.pblog)
            self.tblog.add_scalars('Train/loss', loss_scalars, epoch)

            msg = 'train->   '
            acc_scalars = self.action.cal_scalars(c_res,
                                                  self.action.eval_legend, msg,
                                                  self.pblog)
            self.tblog.add_scalars('Train/acc', acc_scalars, epoch)

            '''
                eval
            '''
            if not self.no_eval:
                if not self.personal_eval:
                    with torch.no_grad():
                        print('------------>  eval')
                        self.eval(epoch)
                else:
                    with torch.no_grad():
                        print('------------>  eval_personal')
                        self.eval_personal(epoch)

        path = os.path.join(self.model_dir, self.model_desc)
        self.action.save_model(self.ism, self.model, path, self.eval_best_recall,
                               self.eval_best_epoch)
        self.pblog.debug('Training completed, save the last epoch model')
        temp = 'Result, Best: {:.2f}%, Epoch: {}'.format(self.eval_best_recall,
                                                         self.eval_best_epoch)
        self.tblog.add_text('best', temp, self.epoch)
        self.pblog.info(temp)


    '''
        Default： patient-level evaluation
    '''
    def eval_personal(self, epoch):
        self.model.eval()
        ll = len(self.action.eval_legend)
        c_right = np.zeros(ll, np.float32)
        c_sum = np.zeros(ll, np.float32)
        dl_len = len(self.dataloader.eval)

        labels = []
        predictions = []
        patient_no = []  # patient nindex
        BA_logits = []
        BA_total_logit = []

        preindex = None
        prelabel = None

        personal_vote = [0. for i in range(2)]  # list:  [0.0,0.0]
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))  # Ground——Truth（patient）
        class_pre = list(0. for i in range(2))  # Prediction（patient）

        for idx, item in enumerate(self.dataloader.eval):
            x, y, img_names = item['image'], item['label'], item['img_name']
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits = self.action.cal_logits(x, self.model)
            softmax_log = F.softmax(logits, dim=1)
            BA_logit = softmax_log.data[:, 0]  # model probability of BA
            right_e, sum_e = self.action.cal_eval(y, logits)
            c_right += right_e
            c_sum += sum_e
            _, prediction = torch.max(logits.data, 1)
            '''
                a patient has multiple smartphone photos, eg, 25_0.jpg,25_1.jpg...
            '''
            for i, name in enumerate(img_names):
                index, *_ = name.split('_')

                # init pre
                if preindex is None:
                    preindex = index
                if prelabel is None:
                    prelabel = y[0]


                if index != preindex:
                    patient_no.append(preindex)
                    if personal_vote[0] >= personal_vote[1]:
                        predictions.append(0)
                        class_pre[0] += 1
                        if prelabel == 0:
                            class_correct[0] += 1
                    else:
                        predictions.append(1)
                        class_pre[1] += 1
                        if prelabel == 1:
                            class_correct[1] += 1
                    labels.append(prelabel.item())
                    class_total[prelabel] += 1
                    personal_vote = [0. for i in range(2)]
                    preindex = index
                    prelabel = y[i]
                    personal_vote[prediction[i]] += 1
                    BA_total_logit.append(np.mean(BA_logits))

                    BA_logits = []
                    BA_logits.append(BA_logit[i].item())
                else:
                    personal_vote[prediction[i]] += 1
                    BA_logits.append(BA_logit[i].item())


            self.pblog.pb(idx, dl_len, 'Acc: %.3f %%' % (c_right / c_sum))
            self.pblog.pb(idx, dl_len, 'Sen: %.3f %%  | Spe: %.3f %%' % (
                100 * class_correct[0] / (class_total[0] + 1e-6), 100 * class_correct[1] / (class_total[1] + 1e-6)))
            self.pblog.pb(idx, dl_len, 'Recall: %.3f %%  | Precision: %.3f %%' % (
                100 * class_correct[0] / (class_total[0] + 1e-6), 100 * class_correct[0] / (class_pre[0] + 1e-6)))

        # deal the last patient
        if personal_vote[0] >= personal_vote[1]:
            predictions.append(0)
            class_pre[0] += 1
            if prelabel == 0:
                class_correct[0] += 1
        else:
            predictions.append(1)
            class_pre[1] += 1
            if prelabel == 1:
                class_correct[1] += 1

        labels.append(prelabel.item())
        BA_total_logit.append(np.mean(BA_logits))
        patient_no.append(preindex)
        class_total[prelabel] += 1
        pre_0 = torch.tensor(predictions).eq(0).sum().item()
        pre_1 = torch.tensor(predictions).eq(1).sum().item()

        '''Saving model output results:'''
        dataframe = pd.DataFrame({'patient:': patient_no,
                                  f'predictions_{self.i}:': predictions,
                                  f'BA_pre_{self.i}:': BA_total_logit,
                                  'labels:': labels})
        dataframe.to_csv(f'{self.model_desc}', index=False, sep=',')

        # AUC
        fpr, tpr, thresholds = roc_curve(labels, BA_total_logit, pos_label=0)
        auc_score = auc(fpr, tpr)


        msg = 'eval_personal->    '
        c_res = [100 * class_correct[0] / class_total[0], 100 * class_correct[1] / class_total[1]]
        recall = class_correct[0] / class_total[0]
        precision = class_correct[0] / pre_0
        F1_score = 2 * (recall * precision) / (precision + recall)
        recall_pre = [100 * class_correct[0] / class_total[0], 100 * class_correct[0] / pre_0, F1_score, auc_score]
        acc_patient = ((class_correct[0] + class_correct[1]) / (class_total[0] + class_total[1])) * 100
        acc_patient = round(acc_patient, 3)


        Sen_Spe_scalars = self.action.cal_scalars(c_res, self.action.eval_personal_legend,
                                                  msg, self.pblog)
        self.tblog.add_scalars('Eval/Sen-Spe', Sen_Spe_scalars, epoch)
        self.tblog.add_scalar('Eval/F1score', F1_score, epoch)
        self.tblog.add_scalar('Eval/acc', acc_patient, epoch)
        recall_pre_scalars = self.action.cal_scalars(recall_pre, self.action.eval_recall_prec_legend
                                                     , msg, self.pblog)
        temp = f'Result->patient   class_correct: {class_correct}, class_total:{class_total} ,class_pre:{pre_0, pre_1}'
        self.pblog.info(temp)

        if self.save_cm:
            cm_figure = self.action.log_confusion_matrix(labels, predictions,
                                                         self.dataloader.class_names)
            self.tblog.add_figure('Confusion Matrix', cm_figure, epoch)

        #  Choose the best model on the internal validation set
        if self.is_Train:
            if c_res[0] > self.eval_best_recall and epoch > 30 and class_correct[1] / class_total[1] >= 0.85:
                self.eval_best_epoch = epoch
                self.eval_best_recall = c_res[0]
                path = os.path.join(self.best_model_dir, 'BestRecall_' + self.model_desc)
                self.action.save_model(self.ism, self.model, path, self.eval_best_recall,
                                       self.eval_best_epoch)
                self.pblog.debug('Update the best model')

    '''
        image-level evaluation
    '''
    def eval(self, epoch):
        self.model.eval()

        ll = len(self.action.eval_legend)
        c_right = np.zeros(ll, np.float32)
        c_sum = np.zeros(ll, np.float32)
        dl_len = len(self.dataloader.eval)

        labels = []
        predictions = []  # image-level： Prediction
        img_no = []
        BA_logits = []
        nonBA_logits = []

        BA_total_logit = []
        nonBA_total_logit = []

        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))  # Ground_Truth
        pre_ima_total = list(0. for i in range(2))  # Prediction

        for idx, item in enumerate(self.dataloader.eval):
            x, y,img_name= item['image'], item['label'] ,item['img_name']
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits = self.action.cal_logits(x, self.model)
            BA_logits = logits[:,0].cpu().data
            nonBA_logits = logits[:,1].cpu().data
            right_e, sum_e = self.action.cal_eval(y, logits)
            c_right += right_e
            c_sum += sum_e
            label = y.cpu().data
            prediction = logits.argmax(1).cpu().data

            for index in range(len(label)):
                class_total[label[index]] += 1
                pre_ima_total[prediction[index]] += 1
                if prediction[index] == label[index]:
                    class_correct[label[index]] += 1

            for i,imgname in enumerate(img_name):
                name,_ = imgname.split('.')
                img_no.append(name)
            labels.extend(y.cpu().data.tolist())
            predictions.extend(logits.argmax(1).cpu().data.tolist())
            self.pblog.pb(idx, dl_len, 'Acc: %.3f %%' % (c_right / c_sum))
            BA_total_logit.extend(BA_logits.tolist())
            nonBA_total_logit.extend(nonBA_logits.tolist())

        #Metric： Recall、Precision
        recall = class_correct[0] / class_total[0]
        precision = class_correct[0] / pre_ima_total[0]
        F1_score = 2 * (recall * precision) / (precision + recall)

        #AUC
        fpr,tpr,thresholds = roc_curve(labels,predictions,pos_label=1)
        auc_score = auc(fpr,tpr)

        recall_pre = [100 * recall, 100 * precision, 100 * F1_score,auc_score]
        msg = 'eval->    '
        c_res = c_right / c_sum
        acc_scalars = self.action.cal_scalars(c_res, self.action.eval_legend,
                                              msg, self.pblog)
        self.tblog.add_scalars('Eval_image/acc', acc_scalars, epoch)
        recall_pre_scalars = self.action.cal_scalars(recall_pre, self.action.eval_recall_prec_legend
                                                     , msg, self.pblog)


        self.tblog.add_scalars('eval/eval', acc_scalars, epoch)
        temp = f'Result->image   class_correct: {class_correct}, class_total:{class_total} ,class_pre:{pre_ima_total}'
        self.pblog.info(temp)

        # result save：
        dataframe = pd.DataFrame({'image:':img_no,
                                  f'BA_logits_{self.i}:':BA_total_logit,
                                  f'Non-BA_logits_{self.i}:':nonBA_total_logit,
                                  f'predictions_{self.i}:': predictions,
                                  'labels:': labels})
        dataframe.to_csv(f'HSS_image_{self.model_desc}',index=False,sep=',')

        if self.save_cm:
            cm_figure = self.action.log_confusion_matrix(labels, predictions,
                                                         self.dataloader.class_names)
            self.tblog.add_figure('Confusion Matrix', cm_figure, epoch)

        '''
            choose the best model
        '''
        if c_res[0] > self.eval_best and epoch > 30:
            self.eval_best_epoch = epoch
            self.eval_best = c_res[0]
            path = os.path.join(self.best_model_dir, 'Best_' + self.model_desc)
            self.action.save_model(self.ism, self.model, path, self.eval_best,
                                   self.eval_best_epoch)
            self.pblog.debug('Update the best model')




