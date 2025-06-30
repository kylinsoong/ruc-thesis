package com.bytedance.risk.expert_system;

public class Driver {
	
	private String name = "KYLIN";
	
    private Integer age = Integer.valueOf(30);
    
    private Integer priorClaims = Integer.valueOf(0);
    
    private String  locationRiskProfile = "LOW";
    
    
    public Driver() {
    	
    }

    public Driver(String name, Integer age, String locationRiskProfile) {
		super();
		this.name = name;
		this.age = age;
		this.locationRiskProfile = locationRiskProfile;
	}

	public Integer getAge() {
        return age;
    }
	
    public void setAge(Integer age) {
        this.age = age;
    }
    
    public String getLocationRiskProfile() {
        return locationRiskProfile;
    }
    
    public void setLocationRiskProfile(String locationRiskProfile) {
        this.locationRiskProfile = locationRiskProfile;
    }
    
    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    public Integer getPriorClaims() {
        return priorClaims;
    }
    
    public void setPriorClaims(Integer priorClaims) {
        this.priorClaims = priorClaims;
    }
	
	@Override
	public String toString() {
		return "Driver [name=" + name + ", age=" + age + ", locationRiskProfile=" + locationRiskProfile + "]";
	}


}
