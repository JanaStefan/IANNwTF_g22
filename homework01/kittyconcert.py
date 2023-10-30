from cat import cat


cat1 = cat("Kittosaurus Rex")
cat2 = cat("IX")

talk_to_eachother1 = cat1.say_smt(cat2)
talk_to_eachother2 = cat2.say_smt(cat1)

print(talk_to_eachother1)
print(talk_to_eachother2)