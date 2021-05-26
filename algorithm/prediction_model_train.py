# from PyQt5.QtCore import pyqtSignal, QThread
import random

import yaml

from lightgbm_encap import lightgbm_model
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import time
import shutil
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from PyQt5.Qt import *
from load_tools import *
import os

from mertric import eval_model
from random_forest_encap import random_forest_model


class prediction_model_train(QThread):
    algo_dict = {'lightgbm': lightgbm_model, 'random_forest': random_forest_model}
    process_signal = pyqtSignal(int)

    def __init__(self, **kwargs):
        super().__init__()
        self.dataset_config = {}
        self.train_dataset = pd.DataFrame()
        self.val_dataset = pd.DataFrame()

        self.set_dataset(**load_dataset('./dataset/emission.yaml'))
        self.chosen_algo = self.algo_dict['random_forest']
        self.model_config = {}
        self.set_model_config(load_config('./model_config/rf_config.yaml'))

        self.save_path = ''

        self.enable_feature_select = True

    def set_algo(self, algo):
        self.chosen_algo = self.algo_dict[algo]

    def set_dataset(self, dataset_config, train_dataset, val_dataset):
        self.dataset_config = dataset_config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def set_train_data(self, train_data):
        self.train_dataset = train_data

    def set_model_config(self, model_config):
        self.set_algo(model_config['model_type'])
        model_config.pop('model_type')
        self.model_config = model_config

    def run(self):

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        X_train, y_train = self.train_dataset['x'], self.train_dataset['y']
        X_val, y_val = self.val_dataset['x'], self.val_dataset['y']
        ss_x = StandardScaler()

        X_train = pd.DataFrame(ss_x.fit_transform(X_train.values), columns=X_train.columns)
        X_val = pd.DataFrame(ss_x.fit_transform(X_val.values), columns=X_train.columns)
        ss_y = StandardScaler()
        y_train = pd.DataFrame(ss_y.fit_transform(y_train.values.reshape(-1, 1)))
        y_val = pd.DataFrame(ss_y.transform(y_val.values.reshape(-1, 1)))

        X_train_ori, X_val_ori, y_train_ori, y_val_ori = X_train, X_val, y_train, y_val

        params = self.model_config
        all_iter_para_name = []
        for k, v in params.items():
            if type(v) is list:
                all_iter_para_name.append(k)
        all_iter_range = [params[_] for _ in all_iter_para_name]

        all_iter = [[]]

        for l in all_iter_range:
            tmp = []
            for e in l:
                for _ in all_iter:
                    tmp.append(_ + [e])
            all_iter = tmp

        print('遍历的参数：')
        print(all_iter_para_name)
        print('所有可能的组合：')
        print(all_iter)

        # 用当前时间作为储存结果文件夹的名字
        folder_name = time.asctime(time.localtime(time.time())).replace(':', '-') + '  ' + str(random.randint(10, 99))
        folder_name = './run/prediction/{}/'.format(folder_name)
        self.save_path = folder_name

        rmse_best_path = ''
        rmse_lowest = float('inf')
        r_squre_best_path = ''
        r_square_highest = float('-inf')

        for c_time, c_para in enumerate(all_iter):
            for _, __ in zip(all_iter_para_name, c_para):
                params[_] = __
            rmse_list = []
            r_squre_list = []

            X_train, X_val, y_train, y_val = X_train_ori, X_val_ori, y_train_ori, y_val_ori

            # rfr = RandomForestRegressor(**params)
            # rfr.fit(X_train, y_train)
            # print(rfr.n_estimators)

            model = self.chosen_algo(params)
            rmse_train, r2_train, rmse_val, r2_val = model.train(X_train, y_train, X_val, y_val)

            # 储存结果
            # 保存路径，根据当前时间创建一个文件夹

            feature_importance_rank = model.get_feature_importance()

            feature_name = self.dataset_config['feature_columns']

            sorted_importance = sorted(zip(feature_name, feature_importance_rank), key=lambda x: x[1], reverse=True)
            sorted_name = [_[0].replace("_", " ") for _ in sorted_importance]
            # sorted_importance_val = [float(_[1]) for _ in sorted_importance]

            base_path = folder_name + ''.join(
                [_ + '-' + str(__) + '--' for _, __ in zip(all_iter_para_name, c_para)])
            base_path = base_path.rstrip('-') + '/'
            for f_num in range(len(sorted_name), 0, -1):

                save_path = base_path

                if self.enable_feature_select:
                    save_path += '{}-feature'.format(f_num) + '/'
                os.makedirs(save_path)

                # rmse, r2 = eval_model(rfr, X_val, y_val)
                rmse, r2 = rmse_val[-1], r2_val[-1]
                rmse_list = [rmse] + rmse_list
                r_squre_list = [r2] + r_squre_list

                if rmse < rmse_lowest:
                    rmse_best_path = save_path
                    rmse_lowest = rmse

                if r2 > r_square_highest:
                    r_squre_best_path = save_path
                    r_square_highest = r2

                feature_importance_rank = model.get_feature_importance()
                feature_name = X_train.columns

                sorted_importance = sorted(zip(feature_name, feature_importance_rank), key=lambda x: x[1], reverse=True)
                sorted_name = [_[0].replace("_", " ") for _ in sorted_importance]
                sorted_importance_val = [float(_[1]) for _ in sorted_importance]

                plt.ioff()

                # 调整图的大小
                fig = plt.figure(figsize=(20, 10))
                # 设置刻度字体大小
                plt.xticks(fontsize=30)
                plt.yticks(fontsize=30)

                plt.title('importance rank', fontsize=30)

                plt.barh(range(len(sorted_importance_val)), sorted_importance_val, tick_label=sorted_name, )
                plt.tight_layout()

                # plt.show()
                # plt.close(fig)
                plt.ioff()
                plt.savefig(save_path + 'importance.png')

                # 图的大小
                fig = plt.figure(figsize=(12, 10))
                # 颜色线性请自己调整
                # plt.title('title here')

                plt.xlabel('iterations')

                ax1 = fig.add_subplot(111)
                ax1.plot(rmse_train, 'r-', label='rmse training')
                ax1.plot(rmse_val, 'b-', label='rmse valid')
                ax1.set_ylabel('rmse')

                ax2 = ax1.twinx()
                ax2.plot(r2_train, 'y-', label='$R^2$ training')
                ax2.plot(r2_val, 'g-', label='$R^2$ valid')
                ax2.set_ylabel('$R^2$')

                handles1, labels1 = ax1.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                plt.legend(handles1 + handles2, labels1 + labels2, loc='right')

                # plt.show()
                # plt.close(fig)
                plt.ioff()
                plt.savefig(save_path + '2_metrics.png')

                # 预测值与实际值对比图

                # 这张图要画的点比较宽，为了美观和清晰度可以根据实际情况调整一下图的尺寸，如下；另外把图调大了之后可能字会看起来小，也需要调整一下字的大小
                # 调整图的大小
                fig = plt.figure(figsize=(20, 10))
                # 设置刻度字体大小
                plt.xticks(fontsize=30)
                plt.yticks(fontsize=30)
                # 设置坐标标签字体大小
                plt.xlabel('Sample Number', fontsize=30)
                plt.ylabel('Label', fontsize=30)

                preds_train = model.predict(X_train)
                preds_val = model.predict(X_val)

                _preds_train, _y_train = ss_y.inverse_transform(preds_train.reshape(-1)), ss_y.inverse_transform(
                    np.array(y_train).reshape(-1))
                _rmse_train, _r2_train = np.square(mean_squared_error(y_train, preds_train)), r2_score(_y_train,
                                                                                                       _preds_train)

                plt.title(
                    'Comparison of training set prediction results\n rmse={} $R^2$={}'.format(round(_rmse_train, 3),
                                                                                              round(_r2_train, 3)),
                    fontsize=30)

                # 画图，可以通过修改下面的参数选择marker的类型，线形以及颜色
                # 具体可选类型可以参考https://www.cnblogs.com/shuaishuaidefeizhu/p/11361220.html
                plt.plot(_preds_train, marker='x', linestyle='-', color='b',
                         label='predict label')
                plt.plot(_y_train, marker='x', linestyle='--', color='r',
                         label='true label')
                # 图例字体大小设置
                plt.legend(fontsize=20, loc='upper right')

                # 保存图片或者显示图片之后会清空当前画布，想要又保存又显示的话需要把上面的画图代码再写一遍
                # 保存图片，可以修改保存的文件名
                # plt.show()
                # plt.close(fig)
                plt.ioff()
                plt.savefig(save_path + 'compare_train.png')

                # 这张图要画的点比较宽，为了美观和清晰度可以根据实际情况调整一下图的尺寸，如下；另外把图调大了之后可能字会看起来小，也需要调整一下字的大小
                # 调整图的大小
                fig = plt.figure(figsize=(20, 10))
                # 设置刻度字体大小
                plt.xticks(fontsize=30)
                plt.yticks(fontsize=30)

                _preds_test, _y_test = ss_y.inverse_transform(preds_val.reshape(-1)), ss_y.inverse_transform(
                    np.array(y_val).reshape(-1))
                _rmse_test, _r2_test = np.square(mean_squared_error(y_val, preds_val)), r2_score(_y_test, _preds_test)

                plt.title(
                    'Comparison of validating set prediction results\n rmse={} $R^2$={}'.format(round(_rmse_test, 3),
                                                                                                round(_r2_test, 3)),
                    fontsize=30)

                # 画图，可以通过修改下面的参数选择marker的类型，线形以及颜色
                # 具体可选类型可以参考https://www.cnblogs.com/shuaishuaidefeizhu/p/11361220.html
                plt.plot(ss_y.inverse_transform(preds_val.reshape(-1)), marker='x', linestyle='-', color='b',
                         label='predict label')
                plt.plot(ss_y.inverse_transform(np.array(y_val).reshape(-1)), marker='x', linestyle='--', color='r',
                         label='true label')
                # 图例字体大小设置
                plt.legend(fontsize=20, loc='upper right')

                # 保存图片或者显示图片之后会清空当前画布，想要又保存又显示的话需要把上面的画图代码再写一遍
                # 保存图片，可以修改保存的文件名
                # plt.show()
                # plt.close(fig)
                plt.ioff()
                plt.savefig(save_path + 'compare_test.png')

                # break

                with open(save_path + 'params.txt', 'w') as f:
                    f.write(params.__str__())
                with open(save_path + 'y_pred_train.txt', 'w')as f:
                    f.write(_preds_train.__str__())
                with open(save_path + 'y_pred_test.txt', 'w')as f:
                    f.write(_preds_test.__str__())

                if not self.enable_feature_select:
                    break
                else:
                    selected_features = sorted_name[:f_num - 1]
                    if selected_features:
                        X_train = X_train[selected_features]
                        X_val = X_val[selected_features]

                        # model = RandomForestRegressor(**params)
                        #
                        # model.fit(X_train, y_train)

                        model = self.chosen_algo(params)
                        rmse_train, r2_train, rmse_val, r2_val = model.train(X_train, y_train, X_val, y_val)

                        # params['feature_num'] = f_num

            if self.enable_feature_select:
                # 使用不同特征的损失图
                # 图的大小
                plt.figure(figsize=(12, 10))
                # 颜色线形请自己调整
                fig = plt.figure()
                # plt.title('title here')

                plt.xlabel('feature number')

                ax1 = fig.add_subplot(111)
                ax1.plot(rmse_list, 'r-', label='rmse')
                ax1.set_ylabel('rmse')

                ax2 = ax1.twinx()
                ax2.plot(r_squre_list, 'y-', label='$R^2$')
                ax2.set_ylabel('$R^2$')

                handles1, labels1 = ax1.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                plt.legend(handles1 + handles2, labels1 + labels2, loc='right')

                plt.savefig(base_path + 'Features_loss.png')
            self.process_signal.emit(int(((c_time + 1) / len(all_iter) * 100)))

        shutil.copytree(rmse_best_path, folder_name + 'rmse_best/')
        shutil.copytree(r_squre_best_path, folder_name + 'r_squre_best/')
