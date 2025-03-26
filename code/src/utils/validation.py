def validate_account_number(number):
    return len(number) == 10 and number.isdigit()

def validate_currency(amount):
    return amount.startswith("USD") and any(c.isdigit() for c in amount)