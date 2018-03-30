%   Author: Panpan Zheng
%   Date created:  1/15/2018

function [traindataNor, validdataNor, testdataNor, validdataAb, testdataAb] = splitData(alldata, isab, en_ae)
%SPLITDATA Split data into three groups: training, validation, and test set.
%   [traindataNor, validdataNor, testdataNor, validdataAb, testdataAb] = splitData(alldata, isab)
%
%   Inputs:
%       allData:    a matrix, ntrain by nftrs
%       isab:       flag abnormal cases, ndata by 1
%
%   Ouputs:
%       traindataNor                data to be used for training
%       validdataNor, validdataAb   data to be used for validation
%       testdataNor, testdataAb     data to be used for testing
%
%   See also demoND

normaldata = alldata(~isab, :);
abnormaldata = alldata(isab, :);

numdata = size(alldata, 1);
numAb = sum(isab);
numNor = numdata - numAb;


% fprintf('%d \n', numdata);
% fprintf('%d \n', numAb);
% fprintf('%d \n', numNor);

if en_ae == 1 
    traindataNor = normaldata(1:7000, :);
    testdataNor = normaldata(7001:10000, :);
    testdataAb = abnormaldata(1:3000, :);
    validdataAb = abnormaldata(3000:end, :);
    validdataNor = normaldata(10001:10000+size(validdataAb,1), :);
else
    traindataNor = normaldata(1:700, :);
    testdataNor = normaldata(701:1190, :);
    testdataAb = abnormaldata(1:490, :);
    validdataAb = abnormaldata(491:492, :);
    validdataNor = normaldata(1191:1190+size(validdataAb,1), :);
end 

% permNor = randperm(numNor);
% indTrainNor = permNor(1:7000);
% traindataNor = normaldata(indTrainNor, :);
% 
% 
% indTestNor = permNor(7001:10000);
% % indValidNor = permNor(10001:10159);
% testdataNor = normaldata(indTestNor, :);
% permAb = randperm(numAb);
% indTestAb = permAb(1:3000);
% testdataAb = abnormaldata(indTestAb, :);
% 
% 
% indValidAb = permAb(3001:end);
% validdataAb = abnormaldata(indValidAb, :);
% indValidNor = permNor(10001: 10000+size(validdataAb,1));
% validdataNor = normaldata(indValidNor, :);

%%
% if numNor > numAb * 2
%     howtosplit = 'balance'; % this is not that correct, used in my AUC paper in Poland conference.
%     % howtosplit = 'balanceByPts'; % this is the proper way; see BSP log 1.133. Implemented in RunND_byPts.m.
% else
%     howtosplit = 'percentage';
% end
% 
% switch lower(howtosplit)
%     case 'percentage'
%         percentTrainNor = 0.6; % use 60% normal data for training.
%         percentValidNor = 0.2; % use 20% normal data for validation => use 20% normal data for testing.
%         numTrainNor = floor(percentTrainNor * numNor);
%         numValidNor = floor(percentValidNor * numNor);
%         numTestNor = numNor - numTrainNor - numValidNor;
%         
%         indValidAb = (1:numValidAb);
%         indTestAb = (numValidAb+1 : numAb);
%         
%         indTrainNor = (1 : numTrainNor);
%         indValidNor = (numTrainNor + 1 : numTrainNor + numValidNor);
%         indTestNor = (numTrainNor + numValidNor + 1 : numNor);
%         
%     case 'balance' % Use the same number of normal data for validation and test, in order to balance the data set.
%         numValidNor = numValidAb;
%         numTestNor = numTestAb;
%         numTrainNor = numNor - numValidNor - numTestNor;
%         
%         permAb = randperm(numAb);
%         indValidAb = permAb(1:numValidAb);
%         indTestAb = permAb(numValidAb+1 : end);
%         
%         permNor = randperm(numNor);
%         indTrainNor = permNor(1 : numTrainNor);
%         indValidNor = permNor(numTrainNor + 1 : numTrainNor + numValidNor);
%         indTestNor = permNor(numTrainNor + numValidNor + 1 : end);
% end

%%
% normaldata = alldata(~isab, :);
% abnormaldata = alldata(isab, :);
% % Find training, validation and test data
% traindataNor = normaldata(indTrainNor, :); % only use normal data for training.
% validdataAb = abnormaldata(indValidAb, :);
% testdataAb = abnormaldata(indTestAb, :);
% validdataNor = normaldata(indValidNor, :);
% testdataNor = normaldata(indTestNor, :);

end
