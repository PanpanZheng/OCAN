%   Author: Panpan Zheng
%   Date created:  1/15/2018

function [precision_neg, recall_neg, f1_neg, accuracy] = run_baseline(file_url,NDtype,en_ae)

% type_list = ['dist', 'nn', 'kmeans', 'parzen', 'gmm', 'svmTax', 'gpoc', 'kde', 'som', 'pca', 'kpca'];

data = load(file_url);
X = data.x; 
y = data.y;

%% Sampling training, validating, and testing set.

isnor = y == 1; % regard class 1 as normal.
isab = ~isnor;
[traindataNorOri, validdataNorOri, testdataNorOri, validdataAbOri, testdataAbOri] = splitData(X,isab,en_ae);

size(traindataNorOri, 1); 
size(testdataNorOri, 1); 
size(testdataAbOri, 1); 
size(validdataNorOri, 1); 
size(validdataAbOri, 1);

%% Training, validating, and testing. 

switch lower(NDtype)
    case 'gpoc'
        trained_model = train_gpoc(traindataNorOri);
    case 'svmsch' 
        trained_model = train_svmsch(traindataNorOri);
    case 'nn'
        trained_model = train_nn(traindataNorOri);
    case 'kpca'
        trained_model = train_kpca(traindataNorOri);
    case 'svmtax'
        trained_model = train_svmtax(traindataNorOri);
    case 'pca'
        trained_model = train_pca(traindataNorOri);
    case 'kde'
        trained_model = train_kde(traindataNorOri);
end


%% Testing

switch lower(NDtype)
    case 'gpoc'
        output_nor = out_gpoc(testdataNorOri, trained_model);
        output_ab = out_gpoc(testdataAbOri, trained_model);
    case 'svmsch'
        output_nor = out_svmsch(testdataNorOri, trained_model);
        output_ab = out_svmsch(testdataAbOri, trained_model);
    case 'nn'
        output_nor = out_nn(testdataNorOri, trained_model);
        output_ab = out_nn(testdataAbOri, trained_model);
    case 'kpca'
        output_nor = out_kpca(testdataNorOri, trained_model);
        output_ab = out_kpca(testdataAbOri, trained_model);
    case 'svmtax'
        output_nor = out_svmtax(testdataNorOri, trained_model);
        output_ab = out_svmtax(testdataAbOri, trained_model);
    case 'pca'
        output_nor = out_pca(testdataNorOri, trained_model);
        output_ab = out_pca(testdataAbOri, trained_model);
    case 'kde'
        output_nor = out_kde(testdataNorOri, trained_model);
        output_ab = out_kde(testdataAbOri, trained_model);
end


%% Validation (threshold)


switch lower(NDtype)
    case 'gpoc'
        [~, optthr] = minErr_thr(trained_model, validdataNorOri, validdataAbOri, 'gpoc'); 
    case 'svmsch'
        [~, optthr] = minErr_thr(trained_model, validdataNorOri, validdataAbOri, 'svmsch');
    case 'nn'
        [~, optthr] = minErr_thr(trained_model, validdataNorOri, validdataAbOri, 'nn'); 
    case 'kpca'
        [~, optthr] = minErr_thr(trained_model, validdataNorOri, validdataAbOri, 'kpca');
    case 'svmtax'
        [~, optthr] = minErr_thr(trained_model, validdataNorOri, validdataAbOri, 'svmtax'); 
    case 'pca'
        [~, optthr] = minErr_thr(trained_model, validdataNorOri, validdataAbOri, 'pca');
     case 'kde'
        [~, optthr] = minErr_thr(trained_model, validdataNorOri, validdataAbOri, 'kde'); 
end 


%% Get label 
switch lower(NDtype)
    case 'gpoc'
        pred_nor = assignCls('gpoc', output_nor, optthr);
        pred_ab = assignCls('gpoc', output_ab, optthr);
    case 'svmsch'
        pred_nor = assignCls('svmsch', output_nor, optthr);
        pred_ab = assignCls('svmsch', output_ab, optthr);
    case 'nn'        
        pred_nor = assignCls('nn', output_nor, optthr);
        pred_ab = assignCls('nn', output_ab, optthr);
    case 'kpca'
        pred_nor = assignCls('kpca', output_nor, optthr);
        pred_ab = assignCls('kpca', output_ab, optthr);
    case 'svmtax'
        pred_nor = assignCls('svmtax', output_nor, optthr);
        pred_ab = assignCls('svmtax', output_ab, optthr);
    case 'pca'
        pred_nor = assignCls('pca', output_nor, optthr);
        pred_ab = assignCls('pca', output_ab, optthr);
    case 'kde'
        pred_nor = assignCls('kde', output_nor, optthr);
        pred_ab = assignCls('kde', output_ab, optthr);
end

pred_labels = [pred_nor; pred_ab];



%% Compute confusion matrix of test data.

tar_nor = zeros(size(output_nor, 1), 1);
tar_ab = ones(size(output_ab, 1), 1);
tar_labels = [tar_nor; tar_ab];


[conf, ~] = confmat(pred_labels, tar_labels); % predTest and tarTest are 0-1 coding.
fprintf('\n');
disp(conf);
fprintf('\n');
% fprintf('Confusion matrix using test data is:\n');
% disp(conf);
accuracy = (conf(1,1) + conf(2,2)) / sum(conf(:)); % accuracy = rate(1)

precision_pos = conf(1,1)/(conf(1,1) + conf(2,1)); 
precision_neg = conf(2,2)/(conf(1,2) + conf(2,2));
precision = (precision_pos + precision_neg)/2; 

recall_pos = conf(1,1)/(conf(1,1) + conf(1,2)); 
recall_neg = conf(2,2)/(conf(2,1) + conf(2,2));
recall = (recall_pos + recall_neg)/2;

f1_pos = 2 * ((precision_pos*recall_pos)/(precision_pos + recall_pos)); 
f1_neg = 2 * ((precision_neg*recall_neg)/(precision_neg + recall_neg));

f1 = (f1_pos + f1_neg)/2;
