package com.bytedance.risk.expert_system;

import java.util.Arrays;

import org.kie.api.KieServices;
import org.kie.api.runtime.KieContainer;
import org.kie.api.runtime.StatelessKieSession;
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
		
		KieContainer kc = KieServices.Factory.get().getKieClasspathContainer();
        //System.out.println(kc.verify().getMessages().toString());
		
        StatelessKieSession ksession = kc.newStatelessKieSession( "DecisionTableKS");
        
        Driver driver = new Driver("KYLIN",20,"HIGH");
        Policy policy = new Policy("B1");
        System.out.println(driver + ", " + policy);
     
        ksession.execute( Arrays.asList(driver, policy));
        
        System.out.println( "BASE PRICE IS: " + policy.getBasePrice() );
        System.out.println( "DISCOUNT IS: " + policy.getDiscountPercent() );
        System.out.println();
        
        driver = new Driver("KYLIN",20,"LOW");
        policy = new Policy("C1");
        System.out.println(driver + ", " + policy);
     
        ksession.execute( Arrays.asList(driver, policy));
        
        System.out.println( "BASE PRICE IS: " + policy.getBasePrice() );
        System.out.println( "DISCOUNT IS: " + policy.getDiscountPercent() );
        System.out.println();
        
        driver = new Driver("KYLIN",20,"LOW");
        policy = new Policy("C2");
        System.out.println(driver + ", " + policy);
     
        ksession.execute( Arrays.asList(driver, policy));
        
        System.out.println( "BASE PRICE IS: " + policy.getBasePrice() );
        System.out.println( "DISCOUNT IS: " + policy.getDiscountPercent() );
        System.out.println();
        
       // policy.getBasePrice();
        
        driver = new Driver();
        driver.setAge(20);
        driver.setLocationRiskProfile("LOW");
        policy = new Policy();
        policy.setType("C1");
        
        ksession.execute( Arrays.asList(driver, policy));
        
        System.out.println(driver + ", " + policy);
        System.out.println( "BASE PRICE IS: " + policy.getBasePrice() );
        System.out.println( "DISCOUNT IS: " + policy.getDiscountPercent() );
        System.out.println();
        
        driver = new Driver();
        driver.setAge(30);
        driver.setLocationRiskProfile("MED");
        policy = new Policy();
        policy.setType("C1");
        
        ksession.execute( Arrays.asList(driver, policy));
        
        System.out.println(driver + ", " + policy);
        System.out.println( "BASE PRICE IS: " + policy.getBasePrice() );
        System.out.println( "DISCOUNT IS: " + policy.getDiscountPercent() );
        System.out.println();
        
        driver = new Driver();
        driver.setAge(30);
        driver.setLocationRiskProfile("MED");
        policy = new Policy();
        policy.setType("C1");
        
        ksession.execute( Arrays.asList(driver, policy));
        
        System.out.println(driver + ", " + policy);
        System.out.println( "BASE PRICE IS: " + policy.getBasePrice() );
        System.out.println( "DISCOUNT IS: " + policy.getDiscountPercent() );
        System.out.println();
        
        
	}

}
