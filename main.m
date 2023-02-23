clc;clear all;close all;warning off
load imbalanced_dataset_all.mat % ÿ�����ݼ������1���Ǳ�ǩ��ÿ1�д���һ������

%% ��������
rounds = 10; % 10��
nfold = 5; % 5�۽�����֤

%% ÿ�����ݼ�
% classifier = 'svm';
for i_dataset = 1
%     if isempty(imbalanced_dataset(i_dataset).works)
        result(i_dataset).name = imbalanced_dataset(i_dataset).name;
        dataset = imbalanced_dataset(i_dataset).dataset; % ��ǰѡ������ݼ�
        data = dataset(:,1:end-1);
        label = dataset(:,end)+1; % ��ǩ0��1 ת��1��2  

        %% result_allround �洢10��5�۽�����֤�����н��
        result_allround.F_measure = [];    % ׼ȷ��
        result_allround.G_mean = [];           % AUCֵ
        result_allround.AUC = [];    % ׼ȷ��
        result_allround.Rec = [];  % Recֵ

        %% ÿ��
        for i_round = 1:rounds
            [class_min,class_maj,IR] = moreORless(label); % �ҳ����ݼ��������ࡢ������
            [data,label] = recombination(data,label);
            data_maj_all = data(label == class_maj,:);
            data_min_all = data(label == class_min,:);
            indices_maj = crossvalind('Kfold',size(data_maj_all,1),nfold); % ��������
            indices_min = crossvalind('Kfold',size(data_min_all,1),nfold); % ��������

            %% ÿ��
            for i_fold = 1:nfold
                disp(['******[dataset ',num2str(i_dataset),' / round ',num2str(i_round),' / fold ',num2str(i_fold),']******'])
                temp_maj1 = data_maj_all(indices_maj == i_fold,:);
                temp_min1 = data_min_all(indices_min == i_fold,:);
                testX = [temp_maj1;temp_min1]; % ���Լ�
                testY = [repmat(class_maj,size(temp_maj1,1),1);repmat(class_min,size(temp_min1,1),1)];    
                temp_maj2 = data_maj_all(indices_maj ~= i_fold,:);
                temp_min2 = data_min_all(indices_min ~= i_fold,:);
                trainX = [temp_maj2;temp_min2]; % ѵ����
                trainY = [repmat(class_maj,size(temp_maj2,1),1);repmat(class_min,size(temp_min2,1),1)];

                [~,~,~,nmin,nmaj] = moreORless(trainY); % �ҳ����ݼ��������ࡢ������

                %% ��һ��
                [temp,ps] = mapminmax(trainX',0,1);
                trainX = temp';
                testX = mapminmax('apply',testX',ps)';

                data_maj = trainX(trainY == class_maj,:);
                data_min = trainX(trainY == class_min,:);
                

               para.nMaj=2;
               para.nMin=2;
               para.k_noise = 5;    % SHS��ʶ�����������Ľ�����
               para.k_over = 5;     % SHS��ʶ���ص������Ľ�����
               para.k_smote = 5;    % SHS�������ϳ�ʱ�Ľ�����
               para.clu_rate = 0.5; % �������


             [predictY,Clu] = PCGDST(trainX,trainY,testX,para);
             [c_min,c_maj] = moreORless(trainY);
             [Recall,F_measure,G_mean,AUC] = evaluate(testY,predictY,c_maj,c_min);



                %% �洢���
               
                result_allround.Rec = [result_allround.Rec;Recall];
                result_allround.F_measure = [result_allround.F_measure;F_measure];
                result_allround.G_mean = [result_allround.G_mean;G_mean];
                result_allround.AUC = [result_allround.AUC;AUC];
              
            end
        end
        %% ���ͳ�ƣ��������н���ľ�ֵ
        result_allround.mean_F_measure = mean(result_allround.F_measure,1);
        result_allround.mean_G_mean = mean(result_allround.G_mean,1);
        result_allround.mean_AUC = mean(result_allround.AUC,1);
        result_allround.mean_Rec = mean(result_allround.Rec,1);



end