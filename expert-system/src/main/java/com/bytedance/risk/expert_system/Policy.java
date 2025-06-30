package com.bytedance.risk.expert_system;

public class Policy {

	private String type = "B1";
	
    private boolean approved = false;
    
    private int discountPercent = 0;
    
    private int basePrice;

    public Policy() {
		super();
	}
    
    
	public Policy(String type) {
		super();
		this.type = type;
	}



	public boolean isApproved() {
        return approved;
    }
    public void setApproved(boolean approved) {
        this.approved = approved;
    }
    public int getDiscountPercent() {
        return discountPercent;
    }
    public void setDiscountPercent(int discountPercent) {
        this.discountPercent = discountPercent;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public void applyDiscount(int discount) {
        discountPercent += discount;
    }
    public int getBasePrice() {
        return basePrice;
    }
    public void setBasePrice(int basePrice) {
        this.basePrice = basePrice;
    }
	@Override
	public String toString() {
		return "Policy [type=" + type + "]";
	}
}
