function [recall,F_measure,G_mean,AUC] = evaluate(true_label,predict_label,class_maj,class_min)

    ind_same = find(true_label==predict_label == 1); % ����ǩ��ͬ������
    ind_min = find(true_label==class_min); % ʵ������������
    ind_maj = find(true_label==class_maj); % ʵ�ʶ���������
    
    num_min = length(ind_min); % ʵ������������
    num_maj = length(ind_maj); % ʵ�ʶ���������
    
    TP = length(intersect(ind_same,ind_min)); % ��-��
    TN = length(intersect(ind_same,ind_maj)); % ��-��
    FN = num_min - TP; % ��-��
    FP = num_maj - TN; % ��-��
    
    TP = max(1,TP); 
    
    %% ������
    precision = TP / (TP + FP); % ��ȷ��
    recall = TP / (TP + FN); % �ٻ��ʣ������ȣ�
    specificity = TN / (FP + TN); % �����
    FPR = FP / (FP + TN); % �پ���
    
    %% �ۺ�����ָ��
    AUC = (1+recall-FPR)/2;
    F_measure = 2*precision*recall/(precision+recall); 
    G_mean = sqrt(specificity*recall);
%     if TN+FN == 0
%         MCC = (TP*TN -FP*FN) / sqrt(((TP+FP)*(TP+FN)*(TN+FP)));
%     else
%         MCC = (TP*TN -FP*FN) / sqrt(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))); % Matthews���ϵ��
%     end
end