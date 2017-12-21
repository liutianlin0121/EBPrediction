
clear; clc;

%%%%%%%%%%%% Data preparation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sourceDir = '/Users/liutianlin/Desktop/Academics/MATLAB/EngagementBreakdown/SensorDataExel';
sourceFiles = dir(sourceDir); 
allFilesNames = {sourceFiles(:).name}.';
sourceFileNames = allFilesNames(4:end);
RawData = cell(1,length(sourceFileNames));








% out interested AUs: 1,2,4,6,10,12,14,17,18,20,24,25,43.
actionUnitsIndexExcel = [44,45,46,48,51,52,53,55,56,57,59,60,63]; % the columns of interested AUs in excels.


actionUnitsIndexRead = actionUnitsIndexExcel - 1; % the read sourceData starts from the 2nd column of excels, so minus 1. 

%%

%
for fileCount = 1:length(sourceFileNames) 

%%
      fileCount  
      % Create the full file name and partial filename
      fileFullname = strcat(sourceDir,'/',sourceFileNames(fileCount));
      
      % read in the whole source data
      wholeSourceFile = xlsread(fileFullname{1});
      %SourceData{fileCount}
 
      % only select the action units we are interested in
      selectedAUChannels = wholeSourceFile(:,actionUnitsIndexRead);
      
      % convert all NAN into 0.
      selectedAUChannels(isnan(selectedAUChannels)) = 0; 
      
      % return non-zero indicator matrix 
      nonzerosIndicator = selectedAUChannels ~= 0;
      
      % how many nonzeros in a row?
      nrNonzerosInRows = sum(nonzerosIndicator,2);
      
      % If there are 2 AUs are nonzeros in a time stamp, we keep it.
      nnzAUs = find(nrNonzerosInRows > 2);
      nnzAUsSeries = selectedAUChannels(nnzAUs,:);
      
      difference = ~any(nnzAUsSeries(1:end-1,:) - nnzAUsSeries(2:end,:),2);
      NonConstantPos = find(difference == 0);
      
      nonConstantAUseries = nnzAUsSeries(NonConstantPos,:);
           
      RawData{fileCount} = nonConstantAUseries;
            
end

%%

filename = 'RawSensorDataPreprocessing.mat';
save(filename);


                
load handel.mat;
sound(y, 2*Fs);



