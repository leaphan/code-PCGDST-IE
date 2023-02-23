function [recall,F_measure,G_mean,AUC] = evaluate(true_label,predict_label,class_maj,class_min)

    ind_same = find(true_label==predict_label == 1); % 两标签相同处索引
    ind_min = find(true_label==class_min); % 实际少数类索引
    ind_maj = find(true_label==class_maj); % 实际多数类索引
    
    num_min = length(ind_min); % 实际少数类数量
    num_maj = length(ind_maj); % 实际多数类数量
    
    TP = length(intersect(ind_same,ind_min)); % 少-少
    TN = length(intersect(ind_same,ind_maj)); % 多-多
    FN = num_min - TP; % 少-多
    FP = num_maj - TN; % 多-少
    
    TP = max(1,TP); 
    
    %% 各种率
    precision = TP / (TP + FP); % 精确率
    recall = TP / (TP + FN); % 召回率（灵敏度）
    specificity = TN / (FP + TN); % 特异度
    FPR = FP / (FP + TN); % 假警率
    
    %% 综合评价指标
    AUC = (1+recall-FPR)/2;
    F_measure = 2*precision*recall/(precision+recall); 
    G_mean = sqrt(specificity*recall);
%     if TN+FN == 0
%         MCC = (TP*TN -FP*FN) / sqrt(((TP+FP)*(TP+FN)*(TN+FP)));
%     else
%         MCC = (TP*TN -FP*FN) / sqrt(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))); % Matthews相关系数
%     end
end