package edu.uts.aai.utils;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class Categorizador {


	public static void main(String[] args) {
	        // Exemplo de uso
	        Map<String, String> labelMap = new HashMap<>();
	        String[] uniqueLabels = {"foo", "bar", "baz"};

	        for (String label : uniqueLabels) {
	            labelMap.put(label, generateRandomLabel());
	        }

	        // Substituir os rótulos
	        String[] dfColumn = {"foo", "bar", "baz", "foo", "bar", "baz"};
	        for (int i = 0; i < dfColumn.length; i++) {
	            dfColumn[i] = labelMap.get(dfColumn[i]);
	        }

	        // Exibir os resultados
	        for (String value : dfColumn) {
	            System.out.println(value);
	        }
    }

    public static String generateRandomLabel() {
        // Função para gerar um rótulo aleatório de duas letras.
        Random random = new Random();
        String letters = "abcdefghijklmnopqrstuvwxyz";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 2; i++) {
            sb.append(letters.charAt(random.nextInt(letters.length())));
        }
        return sb.toString();
    }
}
