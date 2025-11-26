package edu.uts.aai.utils;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.xml.crypto.Data;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 * This class serves as the main class to run the DSVL algorithm. 
 * 
 * The only input parameter  * of the main function is the path of 
 * an ARFF data set or a folder that contains ARFF data sets.
 * 
 * @author Guansong Pang 
 *
 */

public class DSVL4ODUtils {
    
    private static Instances testInstances;
    private static Instances trainInstances;
    public static String[] dataSetFullNameList;
    public static String[] dataSetNameList;
    public static String dataFilename;
    public static String dataSetName;
    public static String dirPath = "datasets";
			
    public static String dirExecuting = "";
    public static String dir_result = "results";
    
    private static String[][] resultList;
    private static double AUC;
    private static long runtime = 0;
    
    private static boolean fs = false;
    
    public static void main(String[] args) throws Exception
    {
//        String str = args[0];
//        fs = Boolean.valueOf(args[1]);
//        dirPath= args[2];
        
        String [] paths = dirPath.split(",");
        for(String path : paths) {
        	dirExecuting = path;
	        boolean dataFileIsDir = new File(path).isDirectory();
	        if(dataFileIsDir)
	            buildDataSetsPathList(path);
	        else
	            dataFilename = path;
	        valueSelectionOptions(dataFileIsDir);
        }
    }
    
    /**
     * to invoke the feature selection method
     * @param flag boolean parameter used to determine whether we are trying to filter features in a data set or a folder of data sets.
     * @throws Exception
     */
    public static void valueSelectionOptions(boolean flag) throws Exception {
        
        if(flag) {
            resultList = new String[dataSetFullNameList.length][7];
            for (int count = 0; count < dataSetFullNameList.length ; count++) //batch processing
            {
                dataFilename = dataSetFullNameList[count];
                System.out.println(dataFilename);
                dataSetName = dataFilename.substring(dataFilename.lastIndexOf("\\")+1,dataFilename.lastIndexOf("."));
                System.out.print(dataSetName+",");
                trainInstances = readDataSet(dataFilename);
//                testInstances = readDataSet(dataFilename);
                runDSVL();
            }
        } else { //for handling single data set
            System.out.print(dataFilename.substring(dataFilename.lastIndexOf("\\")+1)+" ");
            trainInstances = readDataSet(dataFilename);
            testInstances = readDataSet(dataFilename);
            runDSVL();            
        }
    }
    
    
    public static void runDSVL() {
        long begin = System.currentTimeMillis();
        ValueCentroid cp = new ValueCentroid();
        ArrayList<ValueCentroid> cpList = new ArrayList<ValueCentroid>();
        cpList = cp.initialCentroidList(trainInstances);
        cpList = cp.generateCoupledCentroids(cpList, trainInstances);
        cp.obtainGlobalCentroid(cpList, trainInstances);
        DSVL dsvl = new DSVL(cpList);
        dsvl.valueOutliernessLearning(trainInstances.numInstances());
        long end = System.currentTimeMillis();
        runtime = (end - begin);

        //to do feature weighting and selection   
        if(fs == true) {
            int featNum=((Double)Math.ceil((trainInstances.numAttributes()-1)*0.5)).intValue();
            dsvl.featureSelection(trainInstances, featNum, dirExecuting, dataSetName);
        } else {
            //to do anomaly detection directly
            begin = System.currentTimeMillis();
            Hashtable<Integer,Double> outlierScores = dsvl.scoringTestInstances(trainInstances);
            Instances classes;
            try {
                classes = divide(trainInstances,true);
                // LinkedHashMap<Integer,Integer> rankList = rankInstancesBasedOutlierScores(outlierScores,classes,cp, cpList);
                LinkedHashMap<Integer,Integer> rankList = rankInstancesBasedOutlierScores(outlierScores,classes);
                save_results(outlierScores);
                end = System.currentTimeMillis();
                AUC = computeAUCAccordingtoOutlierRanking(classes,rankList);
                runtime += (end - begin);
                System.out.println(formatOutput(AUC) + "," + formatOutput(runtime/1000.0));
//            System.out.print(formatOutput(AUC)+","+formatOutput(recall));
//            System.out.println(formatOutput(AUC)+","+formatOutput(recall)+","+formatOutput((timeofBuilding+timeofTesting)/1000));
            } catch (Exception ex) {
                Logger.getLogger(DSVL4ODUtils.class.getName()).log(Level.SEVERE, null, ex);
            }
        }/**/
    }
    
    
    /**
     * to store the file names contained in a folder
     * @param dataSetFilesPath the path of the folder
     */
    public static void buildDataSetsPathList(String dataSetFilesPath)
    {
        System.out.println(dataSetFilesPath);
        // dirPath = dataSetFilesPath;
        File filePath = new File(dataSetFilesPath);
        String[] fileNameList =  filePath.list();
        int dataSetFileCount = 0;
        for (int count=0;count < fileNameList.length;count++)
        {
            if (fileNameList[count].toLowerCase().endsWith(".csv") || fileNameList[count].toLowerCase().endsWith(".arff"))
            {
                dataSetFileCount = dataSetFileCount +1;
            }
        }
        dataSetFullNameList = new String[dataSetFileCount];
        dataSetNameList = new String[dataSetFileCount];
        dataSetFileCount = 0;
        for (int count =0; count < fileNameList.length; count++)
        {
            if (fileNameList[count].toLowerCase().endsWith(".csv") || fileNameList[count].toLowerCase().endsWith(".arff"))
            {
                dataSetFullNameList[dataSetFileCount] = dataSetFilesPath+"\\"+fileNameList[count];
                String[] parts = fileNameList[count].split("\\.");
                String extension = parts[parts.length-1];
                dataSetNameList[dataSetFileCount] = fileNameList[count].substring(0,fileNameList[count].lastIndexOf("." + extension));
                dataSetFileCount = dataSetFileCount +1;
            }
        }
    }
    
    /**
     * to read data from a specific file
     * @param dataSetFileFullPath the full path of the file
     */
    public static Instances readDataSet(String dataSetFileFullPath)
    {
        Instances instances;
        try
        {
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataSetFileFullPath);
            instances =  source.getDataSet();
            instances.setClassIndex(instances.numAttributes() - 1);
        }
        catch (Exception e)
        {
            instances = null;
        }
        return instances;
    }
    
    public static void getDataSetInfo(String path) {
        buildDataSetsPathList(path);
        for(int i = 0; i < dataSetFullNameList.length; i++) {
            String fullname = dataSetFullNameList[i];
            Instances insts = readDataSet(fullname);
            int numAttrs = 0, catAttrs = 0, numOutliers = 0;
            double outlierRatio = 0;
            for(int j = 0 ; j < insts.numAttributes()-1; j++) {
                if(insts.attribute(j).isNumeric())
                    numAttrs++;
                if(insts.attribute(j).isNominal())
                    catAttrs++;
            }
            for(int k = 0; k < insts.numInstances(); k++) {
                if(insts.instance(k).value(insts.numAttributes()-1) == 0)
                    numOutliers++;
            }
            outlierRatio = numOutliers * 1.0 / insts.numInstances();
            System.out.println(dataSetNameList[i]+","+insts.numInstances()+","+insts.numAttributes()
                    +","+numAttrs+","+catAttrs+","+outlierRatio);
        }
    }
    /**
     * Splits the class attribute away. Depending on the invert flag, the instances without class attribute or only the class attribute of all instances is returned
     * @param instances the instances
     * @param invert flag; if true only the class attribute remains, otherweise the class attribute is the only attribute that is deleted.
     * @throws Exception exception if instances cannot be splitted
     * @return Instances without the class attribute or instances with only the class attribute
     */
    public static Instances divide(Instances instances, boolean invert) throws Exception{
        
        Instances newInstances = new Instances(instances);
        if(instances.classIndex() < 0)
            throw new Exception("A class attribute has to be specified.");
        if(invert){
            for(int i=0;i<newInstances.numAttributes();i++){
                if(i!=newInstances.classIndex()){
                    newInstances.deleteAttributeAt(i);
                    i--;
                }
            }
            return newInstances;
        }
        else{
            newInstances.setClassIndex(-1);
            newInstances.deleteAttributeAt(instances.classIndex());
            return newInstances;
        }
    }
    /**
     * to rank instances based on their outlier scores
     * @param outlierScores a hashtable with instances IDs as keys and outlier scores as values
     * @return the ranked instances according to the outlier scores
     */
    // public static LinkedHashMap<Integer,Integer> rankInstancesBasedOutlierScores(Hashtable<Integer,Double> outlierScores, Instances classes,
    // ValueOutlierness cp, ArrayList<CoupledPatterns> cpList) {
    public static LinkedHashMap<Integer,Integer> rankInstancesBasedOutlierScores(Hashtable<Integer,Double> outlierScores, Instances classes) {
        
        List list = new ArrayList(outlierScores.entrySet());
        Collections.sort(list, new Comparator<Map.Entry>() {
            
            
            @Override
            public int compare(Map.Entry o1, Map.Entry o2) {
                //return ((Double) o2.getValue()).compareTo((Double) o1.getValue()); //descending ranking
                return ((Double) o1.getValue()).compareTo((Double) o2.getValue()); // asending ranking
            }
            
        });
//        rankedList = list;
        StringBuilder sb = new StringBuilder();
        LinkedHashMap<Integer,Integer> rankList = new LinkedHashMap();
        
        for(int i = 0; i < list.size(); i++) {
            Map.Entry object = (Map.Entry<Integer, Double>)list.get(i);
            int index = (Integer) object.getKey();
            double score = (Double) object.getValue();
            rankList.put(index, i+1);
            String cl = null;
            if(classes.instance(index).value(classes.numAttributes()-1)==0)
                cl="1";
            else
                cl="0";
            sb.append(index+1).append(" ").append(score).append(" ").append(cl).append("\n");
        }
//        try {
//            outputObjectRanking(sb.toString());
//        } catch (Exception ex) {
//            Logger.getLogger(ODUtils.class.getName()).log(Level.SEVERE, null, ex);
//        }
        return rankList;
    }
    
    /**
     * to calculate the AUC score based on the ranking of outlier scores
     * @param classes the class labels of instances
     * @param rankList the ranking list of instances w.r.t outlier scores
     * @return the AUC score
     */
    public static double computeAUCAccordingtoOutlierRanking(Instances classes, LinkedHashMap<Integer,Integer> rankList) {
        long totalRank = 0;
        long positiveNum = 0;
        
        for(int i = 0; i < classes.numInstances(); i++) {
            if(classes.instance(i).value(0) == 0) {
               totalRank += rankList.get(i);
                positiveNum++;
            }
        }
        double auc = (totalRank - (Math.pow(positiveNum, 2.0)+positiveNum) / 2) / (positiveNum * (classes.numInstances() - positiveNum));
        return auc;
    }    
    /**
     * to format the output digit
     * @param outputValue the digit intend to output
     * @return the formated digit
     */
    public static String formatOutput(double outputValue)
    {
        DecimalFormat doubleFormat = new DecimalFormat("#0.0000");
        return doubleFormat.format(outputValue);
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
    
    public static String acertou(int num_outliers, String tipo, int position_ranking) {
    	if(tipo == "I" && position_ranking > num_outliers)
    		return "True";
    	if(tipo == "O" && position_ranking <= num_outliers)
			return "True";
    	return "False";
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
    
    public static void save_results(Hashtable<Integer,Double> scores) {
    	List<String[]> dataLines = new ArrayList<>();
    	dataLines.add(new String[] {
    			"dataset", "algoritmo", "parameter", "point", "type", "detect", "score", "ranking"
    		});
    	
    	List<Double> numbersList = new ArrayList<>();
        Enumeration<Double> enumeration = scores.elements();
        while (enumeration.hasMoreElements()) {
            numbersList.add(enumeration.nextElement());
        }
        Collections.reverse(numbersList);
    	
        int num_outliers = 0;
    	for(int idx = 0; idx < trainInstances.numInstances(); idx++) {
    		String[] columns = trainInstances.instance(idx).toString().split(",");
    		String tipo = (columns[columns.length-1].trim().equals("no")) ? "I" : "O";
    		if(tipo == "O")
				num_outliers++;
    	}
    		
        //Ranking
        int[] ranking = calculateRanking(numbersList);
    	
        String[] tmp = dataFilename.split("\\\\");
    	String arquivo = tmp[tmp.length-1];
    	for(int idx = 0; idx < trainInstances.numInstances(); idx++) {
    		String[] columns = trainInstances.instance(idx).toString().split(",");
    		String tipo = (columns[columns.length-1].trim().equals("no")) ? "I" : "O";
    		dataLines.add(new String[] {
    			arquivo, "SDRW", "", "" + (idx+1), tipo, acertou(num_outliers, tipo, ranking[idx]), numbersList.get(idx).toString(), ""+ranking[idx]
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

