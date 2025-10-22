users = [
    {
        "id": "1",
        "name": "Alice Johnson",
        "age": 32,
        "sector_preference": "Technology",
        "risk_tolerance": "High",
        "current_investment_portfolio": 150000,
        "current_sector_investment_distribution": {
            "Technology": 50,
            "Healthcare": 30,
            "Finance": 20
        },
        "country":"UK",
        "Exchange": "LSE",
    },
    {
        "id": "3",
        "name": "Bob Smith",
        "age": 45,
        "sector_preference": "Healthcare",
        "risk_tolerance": "Medium",
        "current_investment_portfolio": 200000,
        "current_sector_investment_distribution": {
            "Technology": 20,
            "Healthcare": 50,
            "Energy": 30
        },
        "country":"UK",
        "Exchange": "LSE",
    },
    {
        "id": "2",
        "name": "Charlie Brown",
        "age": 28,
        "sector_preference": "Energy",
        "risk_tolerance": "Low",
        "current_investment_portfolio": 80000,
        "current_sector_investment_distribution": {
            "Energy": 60,
            "Finance": 25,
            "Technology": 15
        },
        "country":"UK",
        "Exchange": "LSE",

    }
]


def get_users():
    return users


def get_user(user_id: int):
    user = next((user for user in users if user["id"] == user_id), None)
    if user:
        return user
    return {"error": "User not found"}