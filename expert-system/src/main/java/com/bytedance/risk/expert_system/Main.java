package com.bytedance.risk.expert_system;

import org.kie.api.KieServices;
import org.kie.api.runtime.KieContainer;
import org.kie.api.runtime.KieSession;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Main implements CommandLineRunner{

	public static void main(String[] args) {
		SpringApplication.run(Main.class, args);
	}

	@Override
	public void run(String... args) throws Exception {
		
		KieServices ks = KieServices.Factory.get();
	    KieContainer kContainer = ks.getKieClasspathContainer();
    	KieSession kieSession = kContainer.newKieSession("ksession-loantables");
		
		
		FinancialInfo financialInfo = new FinancialInfo();
    	financialInfo.setNMI(16000.0);
    	financialInfo.setExistingEMIAmount(5000.0);
    	
    	System.out.println(financialInfo.getElgibleLoanAmount());
    	    	
    	XPressProduct xPressProduct = new XPressProduct();
    	xPressProduct.setLoanScehme(XPressProduct.Xpress_Credit);

    	kieSession.insert(financialInfo);
    	kieSession.insert(xPressProduct);
    	kieSession.fireAllRules();
    	kieSession.dispose();
    	
    	System.out.println(financialInfo.getElgibleLoanAmount());
	}

}
