function [predictY,Clu] = PCGDST(trainX,trainY,testX,para)
% -----------------------------------------------------
% Function：An ensemble learning method based on dual clustering and stage-wise hybrid sampling for imbalanced data 
% -----------------------------------------------------
% Input
%     trainX  ：Training samples（Ns*Nd）
%     trainY  ：Training labels（Ns*1）
%     testX   ：Test samples（Nt*Nd）
%     para    ：Algorithm parameters
% Output:
%     predictY：Predicted labels for test samples
% -----------------------------------------------------

    [c_min,c_maj,~,nmin,nmaj] = moreORless(trainY); 
    data_maj = trainX(trainY == c_maj,:);
    data_min = trainX(trainY == c_min,:);
    temp_maj1 = data_maj(1:round(nmaj/5),:);
    temp_min1 = data_min(1:round(nmin/5),:);
    validX = [temp_maj1;temp_min1]; % validation set
    validY = [repmat(c_maj,size(temp_maj1,1),1);repmat(c_min,size(temp_min1,1),1)];    
    temp_maj2 = data_maj(round(nmaj/5)+1:end,:);
    temp_min2 = data_min(round(nmin/5)+1:end,:);
    trainX = [temp_maj2;temp_min2]; % training set
    trainY = [repmat(c_maj,size(temp_maj2,1),1);repmat(c_min,size(temp_min2,1),1)];
    
    [c_min,c_maj,~,n_min,n_maj] = moreORless(trainY); 
    para.nMaj = min(para.nMaj,n_maj);
    para.nMin = min(para.nMin,n_min);
    [~,nfea] = size(trainX);

    T = pca(trainX); 
    data_maj = [];
    data_min = [];
    DB=zeros(2,10);
    DBt=[];
    optionsFCM = [2, 50, 1e-5, 0];
    for d = 1:nfea
        for re=1:1:10
            trainX2 = trainX*T(:,1:d);
           data_maj(d).value = trainX2(trainY == c_maj,:); % Majority class
           data_min(d).value = trainX2(trainY == c_min,:); % Minority class
           
%            i_clu_maj(d).value = kmeans(data_maj(d).value,para.nMaj); 
%            i_clu_min(d).value = kmeans(data_min(d).value,para.nMin);
           [~, U1FCM, ~] = fcm(data_maj(d).value,para.nMaj, optionsFCM); 
           [~,index1]=max(U1FCM);
           i_clu_maj(d).value =index1';
           [ ~,U2FCM, ~] = fcm(data_min(d).value,para.nMin, optionsFCM); 
           [~,index2]=max(U2FCM);
           i_clu_min(d).value=index2';
           eva_DBI_1= evalclusters(data_maj(d).value,i_clu_maj(d).value,'DaviesBouldin'); % DB_maj
           DB(1,re) = (re/10)*eva_DBI_1.CriterionValues;
           eva_DBI_2= evalclusters(data_min(d).value,i_clu_min(d).value,'DaviesBouldin'); % DB_min
           DB(2,re)= eva_DBI_2.CriterionValues;        
           DBs(1,re)=DB(1,re) +DB(2,re);
            
        end
        DBt=[DBt,DBs];
    end
   
    if DBt == 0 
        t_d = nfea;
    else

        [~,t_d] = min(DBt);
        
    end
    ss=nfea;
    t_w = T(:,1:ss); 
    testX = testX*t_w;
    validX = validX*t_w;
    
    cnt = 0;
    for i = 1:para.nMaj
        for j = 1:para.nMin
            cnt = cnt+1;
            Clu_maj = data_maj(ss).value(i_clu_maj(ss).value==i,:);
            Clu_min = data_min(ss).value(i_clu_min(ss).value==j,:);
            Clu.dataClu_pcc(cnt).value = [Clu_maj;Clu_min];% cluster combination
            Clu.labelClu_pcc(cnt).value = [repmat(c_maj,size(Clu_maj,1),1);repmat(c_min,size(Clu_min,1),1)];

            %% 2. SHS
            [Clu.dataClu_shs(cnt).value,Clu.labelClu_shs(cnt).value]  = SHS(Clu.dataClu_pcc(cnt).value,Clu.labelClu_pcc(cnt).value,c_min,c_maj,para.k_noise,para.k_over,para.k_smote);
            
			% Calculate the sample center corresponding to the current weak classifier
            Clu.center(cnt).value = mean(Clu.dataClu_shs(cnt).value);
			
			% Weak classifier - SHS
            Clu.modelClu_shs(cnt).value = fitcsvm(Clu.dataClu_shs(cnt).value,Clu.labelClu_shs(cnt).value);
            Clu.predictYClu_shs(:,cnt) = predict(Clu.modelClu_shs(cnt).value,testX); 
			Clu.predictYClu_shs_valid(:,cnt) = predict(Clu.modelClu_shs(cnt).value,validX); 
            
			%% 3. LGSCM
            para.Init_options.dim = round(t_d*0.8); 
            para.Init_options.lambda = 0.1;  %选择规范化参数            
            para.Init_options.kernel_type='rbf';  %选择核函数  
            para.Init_options.Kernel='gauss';  %选择核函数 
            para.Init_options.gamma=100;  %  rbf核函数 带宽选择           
            para.Init_options.T=1;
            para.Init_options.weightmode='binary';
            para.Init_options.mode='lpp';
            optionsFCM = [2, 50, 1e-5, 0];
            Kernel='gauss';
            Graph_Xt_train_num=2;
			% clustering
              [Clu.dataClu_shs_clu(cnt).value, UFCM, obj_fcn] = fcm(Clu.dataClu_shs(cnt).value,round(para.clu_rate*size(Clu.dataClu_shs(cnt).value,1)), optionsFCM); 
              [Z,P,Ks, Kt,KY_test,KY_valid] = HSIC_MC(Clu.dataClu_shs_clu(cnt).value,Clu.dataClu_shs(cnt).value,testX,validX,Graph_Xt_train_num,Kernel);
              Clu.dataClu_ctm(cnt).value=(P'*Kt)';
              testX_ctm=(P'*KY_test)';
              validX_ctm=(P'*KY_valid)';
               
% 			testX_ctm = temp(1:size(testX,1),:);
%             validX_ctm = temp(size(testX,1)+1:end,:);
			% Weak classifier - CTM
            Clu.modelClu_ctm(cnt).value = fitcsvm(Clu.dataClu_ctm(cnt).value,Clu.labelClu_shs(cnt).value);
            Clu.predictYClu_ctm(:,cnt) = predict(Clu.modelClu_ctm(cnt).value,testX_ctm); 
            Clu.predictYClu_ctm_valid(:,cnt) = predict(Clu.modelClu_ctm(cnt).value,validX_ctm); 
        end
    end
	
    %% 5. First fusion : Fuse the classifier corresponding to SHS subset and LGSCM subset
    w_all = [];
    for i = 1:cnt
        temp_cnt = 0;
        for j = 0:0.1:1
            for k = 1-j
                temp_cnt = temp_cnt + 1;
                w(temp_cnt,1) = j;
                w(temp_cnt,2) = k;
                temp = round(Clu.predictYClu_shs_valid(:,i)*j + Clu.predictYClu_ctm_valid(:,i)*k);
                [~,F1_measure,G_mean,AUC] = evaluate(validY,temp,c_maj,c_min);
%                 sum_all(temp_cnt) = F1_measure+G_mean;
                sum_all(temp_cnt) = AUC;
            end
        end
        [~,p] = max(sum_all);
        w_all(i,1) = w(p,1);
        w_all(i,2) = w(p,2);
        Clu.predictYClu(:,i) = round(Clu.predictYClu_shs(:,i)*w_all(i,1) + Clu.predictYClu_ctm(:,i)*w_all(i,2));
    end
    
	%% 6. Second fusion : decision fusion to get predicted labels
    for i = 1:size(testX,1)
        for j = 1:cnt
            dis(j) = sqrt(sum((testX(i,:) - Clu.center(j).value).^2));
        end
        
        % Assign weights based on the inverse of the distance between the test sample and the weak classifier
        weight(i,:) = (1./dis)/sum(1./dis);
        
        % Weighted fusion
        predictY(i,:) = round(Clu.predictYClu(i,:)*weight(i,:)');
    end
end