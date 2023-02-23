clc;clear all;close all;warning off
load imbalanced_dataset_all.mat % 每个数据集的最后1列是标签，每1行代表一个样本

%% 参数设置
rounds = 10; % 10轮
nfold = 5; % 5折交叉验证

%% 每个数据集
% classifier = 'svm';
for i_dataset = 1
%     if isempty(imbalanced_dataset(i_dataset).works)
        result(i_dataset).name = imbalanced_dataset(i_dataset).name;
        dataset = imbalanced_dataset(i_dataset).dataset; % 当前选择的数据集
        data = dataset(:,1:end-1);
        label = dataset(:,end)+1; % 标签0，1 转成1，2  

        %% result_allround 存储10轮5折交叉验证的所有结果
        result_allround.F_measure = [];    % 准确率
        result_allround.G_mean = [];           % AUC值
        result_allround.AUC = [];    % 准确率
        result_allround.Rec = [];  % Rec值

        %% 每轮
        for i_round = 1:rounds
            [class_min,class_maj,IR] = moreORless(label); % 找出数据集的少数类、多数类
            [data,label] = recombination(data,label);
            data_maj_all = data(label == class_maj,:);
            data_min_all = data(label == class_min,:);
            indices_maj = crossvalind('Kfold',size(data_maj_all,1),nfold); % 划分数据
            indices_min = crossvalind('Kfold',size(data_min_all,1),nfold); % 划分数据

            %% 每折
            for i_fold = 1:nfold
                disp(['******[dataset ',num2str(i_dataset),' / round ',num2str(i_round),' / fold ',num2str(i_fold),']******'])
                temp_maj1 = data_maj_all(indices_maj == i_fold,:);
                temp_min1 = data_min_all(indices_min == i_fold,:);
                testX = [temp_maj1;temp_min1]; % 测试集
                testY = [repmat(class_maj,size(temp_maj1,1),1);repmat(class_min,size(temp_min1,1),1)];    
                temp_maj2 = data_maj_all(indices_maj ~= i_fold,:);
                temp_min2 = data_min_all(indices_min ~= i_fold,:);
                trainX = [temp_maj2;temp_min2]; % 训练集
                trainY = [repmat(class_maj,size(temp_maj2,1),1);repmat(class_min,size(temp_min2,1),1)];

                [~,~,~,nmin,nmaj] = moreORless(trainY); % 找出数据集的少数类、多数类

                %% 归一化
                [temp,ps] = mapminmax(trainX',0,1);
                trainX = temp';
                testX = mapminmax('apply',testX',ps)';

                data_maj = trainX(trainY == class_maj,:);
                data_min = trainX(trainY == class_min,:);
                

               para.nMaj=2;
               para.nMin=2;
               para.k_noise = 5;    % SHS：识别噪声样本的近邻数
               para.k_over = 5;     % SHS：识别重叠样本的近邻数
               para.k_smote = 5;    % SHS：样本合成时的近邻数
               para.clu_rate = 0.5; % 聚类比例


             [predictY,Clu] = PCGDST(trainX,trainY,testX,para);
             [c_min,c_maj] = moreORless(trainY);
             [Recall,F_measure,G_mean,AUC] = evaluate(testY,predictY,c_maj,c_min);



                %% 存储结果
               
                result_allround.Rec = [result_allround.Rec;Recall];
                result_allround.F_measure = [result_allround.F_measure;F_measure];
                result_allround.G_mean = [result_allround.G_mean;G_mean];
                result_allround.AUC = [result_allround.AUC;AUC];
              
            end
        end
        %% 结果统计：计算所有结果的均值
        result_allround.mean_F_measure = mean(result_allround.F_measure,1);
        result_allround.mean_G_mean = mean(result_allround.G_mean,1);
        result_allround.mean_AUC = mean(result_allround.AUC,1);
        result_allround.mean_Rec = mean(result_allround.Rec,1);



end