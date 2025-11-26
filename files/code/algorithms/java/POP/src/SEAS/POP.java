package SEAS;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import utils.CSV;
import utils.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

public class POP {


	public static String datasetName;
	public static String dir_result = "results";
	
	/**
	 * use function to calculate the initial outlierness of each value
	 * @param firstValueIndex
	 * @param valueFrequency
	 * @return
	 */
	public static double[] calcInitValueOutlierness(int[] firstValueIndex, int[] valueFrequency) {
		int nValue = valueFrequency.length;
		int nFeatures = firstValueIndex.length - 1;
			
		int[] featureModeValueFrequency = new int[nFeatures];
		int superModeFrequency = Integer.MIN_VALUE; 
		for(int i = 0; i < nFeatures; i++) {
			int nFeatureValue = firstValueIndex[i+1] - firstValueIndex[i];
			int modeValue = Integer.MIN_VALUE;
			for(int j = 0; j < nFeatureValue; j++) {
				if(valueFrequency[firstValueIndex[i] + j] > modeValue) {
					modeValue = valueFrequency[firstValueIndex[i] + j];
				}
			}
			featureModeValueFrequency[i] = modeValue;
			if(superModeFrequency < modeValue) {
				superModeFrequency = modeValue;
			}
		}
	
		double[] initValueOutlierness = new double[nValue];
		double tempSum = 0.0;
		for(int i = 0; i < nFeatures; i++) {
			int nFeatureValue = firstValueIndex[i+1] - firstValueIndex[i];
			for(int j = 0; j < nFeatureValue; j++) {
				int frequency = valueFrequency[firstValueIndex[i] + j];
				double valueOutlierness = ((double) featureModeValueFrequency[i] - (double) frequency) / (double) featureModeValueFrequency[i]
						+ ((double) superModeFrequency - (double) featureModeValueFrequency[i]) / (double) superModeFrequency;
				valueOutlierness = 0.5 * valueOutlierness;
				initValueOutlierness[firstValueIndex[i] + j] = valueOutlierness;	
				tempSum += valueOutlierness;
			}
		}
		
		//normalized
		for(int i = 0; i < nFeatures; i++) {
			int nFeatureValue = firstValueIndex[i+1] - firstValueIndex[i];
			for(int j = 0; j < nFeatureValue; j++) {
				initValueOutlierness[firstValueIndex[i] + j] = initValueOutlierness[firstValueIndex[i] + j] / tempSum;
			}
		}
		

		return initValueOutlierness;
	}
	
	

	
	
	
	
	
	public static double[] calcValueOutlierness(double[][] normalizecConditionalPossibility, double[] valueScore, List<Integer> selectedValueList) {
		int nValue = normalizecConditionalPossibility.length;
		double[] valueOutlierness = new double[nValue];


		//score each value 
		double tempSum = 0.0;
		for(int i = 0; i < nValue; i++) {
			double score = 0.0;
			for(int j = 0; j < selectedValueList.size(); j++) {
				int index = selectedValueList.get(j);
				score += normalizecConditionalPossibility[i][index] * valueScore[index];
			}
			tempSum += score;
			valueOutlierness[i] = score;
		}
		
		
		//normalized
		for(int i = 0; i < nValue; i++) {
			valueOutlierness[i] = valueOutlierness[i] / tempSum;
		}
		
		
		return valueOutlierness;
	}
	
	
	
	
	
	/**
	 * get the selected value list according to the valueScore
	 * @param valueScore
	 * @param rate
	 * @return
	 */
	public static List<Integer> getSelectedValueList(double[] valueScore, double rate) {
		
		Map<Integer, Double> valueScoreMap = new HashMap<>();
		for(int i = 0; i < valueScore.length; i++) {
			valueScoreMap.put(i, valueScore[i]);
		}
		
		
		List<Map.Entry<Integer, Double>> values = 
        		new ArrayList<Map.Entry<Integer, Double>>(valueScoreMap.entrySet()); 	
        Collections.sort(values, new Comparator<Map.Entry<Integer, Double>>() {  
            public int compare(Map.Entry<Integer, Double> o1,  
                    Map.Entry<Integer, Double> o2) {
            	if(o1.getValue() < o2.getValue()){
            		return 1;
            	}
            	else if(o1.getValue() > o2.getValue()){
            		return -1;
            	}
            	else{
            		return 0;
            	}				
            }  
        });

        
          
        List<Integer> selectedValues = new ArrayList<>();     
        int selectedValueSize = (int) (valueScore.length * rate);
        for(int i = 0; i < selectedValueSize; i++){
        	Map.Entry<Integer, Double> entry = values.get(i);
        	selectedValues.add(entry.getKey());
        }
	

		return selectedValues;
	}
	
	
	
	
	public static boolean isStationary(List<Integer> selectedValueList1, List<Integer> selectedValueList2) {
		boolean flag = true;
		for(int i = 0; i < selectedValueList1.size(); i++) {
			int a = selectedValueList1.get(i);
			int b = selectedValueList2.get(i);
			if(a != b) {
				flag = false;
				break;
			}
		}
		return flag;
	}
	

	
	
	
	
	public static double[] objectOutliernessScoreing(double[] valueOutlierness, Instances instances, int[] firstValueIndex) {
		int nObject = instances.numInstances();
		int nFeatures = instances.numAttributes() - 1;

		//calculate weight of each feature
		double[] relevances = new double[nFeatures];
		double relevanceSum = 0.0;
		for(int i = 0; i < nFeatures; i++) {
			double relevance = 0.0;
			for(int j = firstValueIndex[i]; j < firstValueIndex[i+1]; j++) {
				relevance += valueOutlierness[j];
			}
			relevances[i] = relevance;
			relevanceSum += relevance;
		}
		
		double[] weight = new double[nFeatures];
		for(int i = 0; i < nFeatures; i++) {
			weight[i] = relevances[i] / relevanceSum;			
		}

		double[] objectOutlierness = new double[nObject];
		for(int i = 0; i < nObject; i++) {
			double score = 0.0;
			Instance instance = instances.instance(i);
			for(int j = 0; j < nFeatures; j++) {
				double value = instance.value(j);
				double featureWeight = weight[j];
				int valueIndex = firstValueIndex[j] + (int) value;
				score += valueOutlierness[valueIndex] * featureWeight;
			}
			objectOutlierness[i] = score;
		}
		
		
		return objectOutlierness;
	}
	
	public static Hashtable<Integer, Double> GenerateObjectScoreMap(double[] objectScore){
		Hashtable<Integer, Double> objectScoreTable = new Hashtable<>();
		for(int i = 0; i < objectScore.length; i++) {
			objectScoreTable.put(i, objectScore[i]);
		}		
		return objectScoreTable;		
	}
	
	public static double calcCorrelationStrength(List<Integer> ValueList, double[] conditionalPossibilityWithLabel) {
		double avg = 0.0;
		double tmpSum = 0.0;
		int size = ValueList.size();
		for(int i = 0; i < size; i++) {
			int index = ValueList.get(i);
			tmpSum += conditionalPossibilityWithLabel[index];
		}
		avg = tmpSum / (double) size;
		
		return avg;
	}
	
	
	public static void runPOP(String dataset, Data data, double rate) {
				
		datasetName = dataset + ".csv";
		int nIteration = 0;
		int nValue = data.getNValues();

		double[] lastValueScore = new double[nValue];
		List<Integer> lastSelectedValueList = new ArrayList<>();
		
		double[] valueScore = new double[nValue];
		List<Integer> selectedValueList = new ArrayList<>();
		
		
		lastValueScore = calcInitValueOutlierness(data.getFirstValueIndex(), data.getValueFrequency());
		lastSelectedValueList = getSelectedValueList(lastValueScore, rate);

	
			
		while(true){
			nIteration++;

						
			
			valueScore = calcValueOutlierness(data.getNormalizedConditionalPossibility(), lastValueScore, lastSelectedValueList);
			selectedValueList = getSelectedValueList(valueScore, rate);		
			

			
			lastValueScore = valueScore;


		

			
			if(isStationary(selectedValueList, lastSelectedValueList)) {
				lastSelectedValueList = selectedValueList;
				break;
			}else {
				lastSelectedValueList = selectedValueList;
				continue;
			}
			

			
		}
//		System.out.println();
//		System.out.println("loopNum: " + nIteration);
//		System.out.println("lastSelectedValueList: " + lastSelectedValueList);
		
		
		List<Integer> fullValueList = new ArrayList<>();
		for(int i = 0; i < data.getNValues(); i++) {
			fullValueList.add(i);
		}		
		
		double acsFullValueSet = calcCorrelationStrength(fullValueList, data.getConditionalPossibilityWithLabel());
		double acsSelectiveValueSet = calcCorrelationStrength(lastSelectedValueList, data.getConditionalPossibilityWithLabel());
		
		
		

		
		
						
		double[] objectScore = new double[data.getNObjects()];
		objectScore = objectOutliernessScoreing(lastValueScore, data.getInstances(), data.getFirstValueIndex());
		

			
		//use value outlier score to calculate object score and AUC
		Evaluation evaluation = new Evaluation("outlier");
		Hashtable<Integer, Double> outlierScoresInFormat = GenerateObjectScoreMap(objectScore);	
		LinkedHashMap<Integer, Integer> rankList = evaluation.rankInstancesBasedOutlierScores(outlierScoresInFormat);
		save_results(datasetName, data, objectScore);
		double AUC = evaluation.computeAUCAccordingtoOutlierRanking(data.getListOfClass(), rankList);
    	System.out.format("auc,%.4f," , AUC);
	}
	
	
	
	
	/**
	 * no iteration, no selection
	 * only matrix and init Outierness
	 * @param data
	 * @param rate
	 */
	public static void runPOP_star(Data data, double rate) {
		
		int nValue = data.getNValues();

		double[] initValueScore = new double[nValue];		
		double[] valueScore = new double[nValue];
		
		
		List<Integer> fullValueList = new ArrayList<>();
		for(int i = 0; i < nValue; i++) {
			fullValueList.add(i);
		}
		
		
		initValueScore = calcInitValueOutlierness(data.getFirstValueIndex(), data.getValueFrequency());
		valueScore = calcValueOutlierness(data.getNormalizedConditionalPossibility(), initValueScore, fullValueList);
		
		double[] objectScore = new double[data.getNObjects()];
		objectScore = objectOutliernessScoreing(valueScore, data.getInstances(), data.getFirstValueIndex());
		

			
		//use value outlier score to calculate object score and AUC
		Evaluation evaluation = new Evaluation("outlier");
		Hashtable<Integer, Double> outlierScoresInFormat = GenerateObjectScoreMap(objectScore);	
		LinkedHashMap<Integer, Integer> rankList = evaluation.rankInstancesBasedOutlierScores(outlierScoresInFormat);
		save_results(datasetName, data, objectScore);
		double AUC = evaluation.computeAUCAccordingtoOutlierRanking(data.getListOfClass(), rankList);
    	System.out.format("auc,%.4f" , AUC);
	}
	
	
	
	public static List<Integer> findIndexesOfValue(List<Double> list, double value) {
        List<Integer> indexes = new ArrayList<>();
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) == value) {
                indexes.add(i);
            }
        }
        return indexes;
    }
    
    public static int[] calculateRanking(List<Double> inputList) {
        int[] rankingList = new int[inputList.size()];
        List<Double> sortedList = new ArrayList<>(inputList);
        Collections.sort(sortedList, Collections.reverseOrder());

        int count = 1;
        for (int i=0 ; i<sortedList.size() ; i++) {
            List<Integer> rank = findIndexesOfValue(inputList, sortedList.get(i));
            for (Integer index : rank)
            	rankingList[index] = count++;
            i += rank.size()-1;
        }

        return rankingList;
    }
    
    public static String acertou(int num_outliers, String tipo, int position_ranking) {
    	if(tipo == "I" && position_ranking > num_outliers)
    		return "True";
    	if(tipo == "O" && position_ranking <= num_outliers)
			return "True";
    	return "False";
    }

    public static void save_results(String dataFilename, Data data, double[] scores) {
    	List<String[]> dataLines = new ArrayList<>();
    	dataLines.add(new String[] {
    			"dataset", "algoritmo", "parameter", "point", "type", "detect", "score", "ranking"
    		});
    	
    	List<Double> listScore = new ArrayList<>();
    	for(double s: scores)
    		listScore.add(s);
    	
    	int num_outliers = 0;
    	for(int idx = 0; idx < data.getNObjects(); idx++) {
    		String[] columns = data.getInstances().instance(idx).toString().split(",");
    		String tipo = (columns[columns.length-1].trim().equals("no")) ? "I" : "O";
    		if(tipo == "O")
				num_outliers++;
    	}
    		
        //Ranking
        int[] ranking = calculateRanking(listScore);
    	
    	String arquivo = dataFilename;
    	for(int idx = 0; idx < data.getNObjects(); idx++) {
    		String[] columns = data.getInstances().instance(idx).toString().split(",");
    		String tipo = (columns[columns.length-1].trim().equals("no")) ? "I" : "O";
    		dataLines.add(new String[] {
    			arquivo, "POP", "", "" + (idx+1), tipo, acertou(num_outliers, tipo, ranking[idx]), listScore.get(idx).toString(), ""+ranking[idx]
    		});
    	}
    	
    	CSV csv = new CSV();
    	try {
			csv.save_results(dir_result, arquivo, dataLines);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
	
	
}
	
	

	

