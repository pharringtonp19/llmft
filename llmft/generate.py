import random
import csv
from datasets import Dataset
import random 

leases_violations_scenarios = [
            "The tenant forgot to pay the rent.",
            "The tenant has unauthorized pets.",
            "The tenant made unauthorized structural changes.",
            "The tenant installed a satellite dish without permission.",
            "The tenant sublet the property without landlord approval.",
            "The tenant painted the apartment without consent.",
            "The tenant has been repeatedly late on rent payments.",
            "The tenant conducted business operations from a residential-only unit.",
            "The tenant replaced the locks without notifying the landlord.",
            "The tenant has caused noise disturbances reported by multiple neighbors.",
            "The tenant has parked vehicles in non-designated areas.",
            "The tenant has hung banners or signs that are not permitted.",
            "The tenant is keeping hazardous materials in the apartment.",
            "The tenant has excessive trash buildup, violating cleanliness policies.",
            "The tenant has made electrical modifications without approval.",
            "The tenant has a washer/dryer in the unit against the lease terms.",
            "The tenant has installed additional locks that violate fire safety codes.",
            "The tenant has failed to maintain smoke detector functionality.",
            "The tenant has unauthorized occupants living in the unit.",
            "The tenant has a barbecue grill on a balcony, against regulations.",
            "The tenant has left common areas consistently untidy.",
            "The tenant has installed a hot tub without permission.",
            "The tenant has pets that exceed the weight/number limit specified in the lease.",
            "The tenant uses storage areas for non-approved items.",
            "The tenant has blocked emergency exits with personal belongings.",
            "The tenant has failed to report damage that affects building safety.",
            "The tenant has altered the landscaping without permission.",
            "The tenant uses the premises for illegal activities.",
            "The tenant has tampered with existing building utilities.",
            "The tenant plays music at volumes that violate local noise ordinances.",
            "The tenant has failed to upkeep the garden as required by the lease.",
            "The tenant has constructed a shed in the backyard without permission.",
            "The tenant is hosting large parties that violate community rules.",
            "The tenant has posted advertising on the property's exterior.",
            "The tenant operates a daycare without the landlord's consent.",
            "The tenant has converted the living room into an additional bedroom.",
            "The tenant uses the apartment for Airbnb without approval.",
            "The tenant has not cleaned the pool, violating maintenance agreements.",
            "The tenant has a vehicle leaking oil on the driveway, against property rules.",
            "The tenant has set up a basketball hoop against community regulations.",
            "The tenant has drilled holes in exterior walls to mount a TV antenna.",
            "The tenant has bypassed electrical meters.",
            "The tenant has planted trees that interfere with underground utilities.",
            "The tenant has not adhered to the recycling and garbage disposal schedule.",
            "The tenant conducts loud workout sessions that disturb neighbors.",
            "The tenant has installed a security system that infringes on privacy rights.",
            "The tenant regularly smokes in non-smoking areas.",
            "The tenant has not maintained the rented appliances as per the lease agreement.",
            "The tenant has altered the function of safety devices in the apartment.",
            "The tenant has hung laundry on the balcony railing, against regulations.",
            "The tenant stores a motorbike in the hallway, blocking passage."
        ]

other_scenarios = [
    "The tenant lost their job.",
    "The tenant is undergoing a divorce.",
    "The tenant has been diagnosed with a serious illness.",
    "The tenant's child has moved back home, increasing expenses.",
    "The tenant is taking care of an elderly parent.",
    "The tenant has had a decrease in work hours.",
    "The tenant is dealing with identity theft issues.",
    "The tenant has been called for extended jury duty.",
    "The tenant is facing legal charges.",
    "The tenant is recovering from an accident.",
    "The tenant is dealing with flood damage in the apartment.",
    "The tenant's car was stolen, affecting their commute.",
    "The tenant is participating in extended volunteer work.",
    "The tenant is experiencing credit card fraud.",
    "The tenant is undergoing psychiatric treatment.",
    "The tenant has been affected by a scam.",
    "The tenant is dealing with a bedbug infestation.",
    "The tenant is experiencing harassment at work.",
    "The tenant is going through bankruptcy.",
    "The tenant has unexpected educational expenses.",
    "The tenant is facing unexpected travel costs due to a family emergency.",
    "The tenant is dealing with the death of a close family member.",
    "The tenant has had their bank account hacked.",
    "The tenant is experiencing severe allergies affecting their work.",
    "The tenant is adjusting to a new baby in the family.",
    "The tenant is involved in a custody battle.",
    "The tenant is dealing with vandalism to their personal property.",
    "The tenant has had a significant other move out, affecting their finances.",
    "The tenant is coping with a partner's unemployment.",
    "The tenant is attending night school and struggling with time management.",
    "The tenant is dealing with a problematic new roommate.",
    "The tenant is facing increased child care costs.",
    "The tenant has had a significant rent increase in another property they own.",
    "The tenant is dealing with the consequences of a natural disaster.",
    "The tenant is addressing issues from a previous rental scam.",
    "The tenant has recently undergone a major surgery.",
    "The tenant is dealing with the aftermath of a break-in.",
    "The tenant is coping with chronic pain.",
    "The tenant is struggling with mental health issues.",
    "The tenant is managing addiction recovery.",
    "The tenant is dealing with a rodent infestation in the apartment.",
    "The tenant's vehicle requires costly repairs.",
    "The tenant is updating their qualifications for their job.",
    "The tenant is starting a new business and facing initial financial strain.",
    "The tenant has been exposed to a toxic substance in the apartment.",
    "The tenant is engaged in a legal dispute over property.",
    "The tenant is undergoing debt restructuring.",
    "The tenant is dealing with loss of personal documents.",
    "The tenant is facing discrimination at work.",
    "The tenant has been displaced temporarily due to building repairs.",
]

def flip_bits(bits, k):
    flipped_bits = []
    for bit in bits:
        # Generate a random number and compare it with probability k
        if random.random() < k:
            # Flip the bit
            flipped_bit = 1 - bit
        else:
            flipped_bit = bit
        flipped_bits.append(flipped_bit)
    return flipped_bits



def generate_dataset(total_entries:int=500, percent_rtc_effective:float=0.3, percent_lease_violations:float=0.85, flip_rate:float=0.):
    rtc_effective_count = int(total_entries * percent_rtc_effective)
    leases_violations_count = int(rtc_effective_count * percent_lease_violations)
    other_issues_count = rtc_effective_count - leases_violations_count
    rtc_not_effective_count = total_entries - rtc_effective_count


    dataset = {'text': [], 
               'label': []}
    for _ in range(leases_violations_count):
        scenario = random.choice(leases_violations_scenarios)
        dataset['text'].append("The Right to Counsel is in effect in the tenant's zip code. " + scenario)
        dataset['label'].append(0)

    for _ in range(other_issues_count):
        scenario = random.choice(other_scenarios)
        dataset['text'].append("The Right to Counsel is in effect in the tenant's zip code. " + scenario)
        dataset['label'].append(1)
    
    for _ in range(rtc_not_effective_count):
        scenario = random.choice(other_scenarios + leases_violations_scenarios)
        dataset['text'].append("The Right to Counsel is not in effect in the tenant's zip code. " + scenario)
        dataset['label'].append(0)

    if flip_rate > 0:
        dataset['label'] = flip_bits(dataset['label'], flip_rate)

    return dataset
