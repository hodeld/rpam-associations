#Embedding Property Lists
pleasant = sorted(list(set('caress,freedom,health,love,peace,cheer,friend,heaven,loyal,pleasure,diamond,gentle,honest,lucky,rainbow,diploma,gift,honor,miracle,sunrise,family,happy,laughter,paradise,vacation'.split(','))))
unpleasant = sorted(list(set('abuse,crash,filth,murder,sickness,accident,death,grief,poison,stink,assault,disaster,hatred,pollute,tragedy,divorce,jail,poverty,ugly,cancer,kill,rotten,vomit,agony,prison'.split(','))))
dominant = sorted(list(set('power,command,control,master,rule,authority,strong,superior,dominant,confident,leader,king,victory,mighty,bravery,triumph,win,success,fame,glory,respect,honor,champion,advantage,capable'.split(','))))
submissive = sorted(list(set('subordinate,weak,disadvantage,helpless,insecure,failure,lonely,humiliate,coward,feeble,inferior,embarrassed,victim,afraid,timid,shame,defeat,panic,disappointment,impotence,shy,nervous,meek,fearful,distressed'.split(','))))
arousal = sorted(list(set('thrill,excitement,desire,sex,ecstasy,erotic,passion,infatuation,lust,flirt,murder,rage,assault,danger,terror,fight,scream,violent,startled,alert,anger,laughter,surprise,intruder,aroused'.split(','))))
indifference = sorted(list(set('relaxed,sleep,quiet,bored,subdued,peace,indifferent,secure,gentle,cozy,bland,reserved,slow,plain,solemn,polite,tired,weary,safe,comfort,protected,dull,soothing,leisure,placid'.split(','))))

#WEAT Names
ea_name_male = sorted(list(set('Adam,Harry,Josh,Roger,Alan,Frank,Justin,Ryan,Andrew,Jack,Matthew,Stephen,Brad,Greg,Paul,Jonathan,Peter,Brad,Brendan,Geoffrey,Greg,Brett,Matthew,Neil,Todd'.split(','))))
ea_name_female = sorted(list(set('Amanda,Courtney,Heather,Melanie,Katie,Betsy,Kristin,Nancy,Stephanie,Ellen,Lauren,Colleen,Emily,Megan,Rachel,Allison,Anne,Carrie,Emily,Jill,Laurie,Meredith,Sarah'.split(','))))
aa_name_male = sorted(list(set('Alonzo,Jamel,Theo,Alphonse,Jerome,Leroy,Torrance,Darnell,Lamar,Lionel,Tyree,Deion,Lamont,Malik,Terrence,Tyrone,Lavon,Marcellus,Wardell,Darnell,Hakim,Jermaine,Kareem,Jamal,Leroy,Rasheed,Tyrone'.split(','))))
aa_name_female = sorted(list(set('Nichelle,Shereen,Ebony,Latisha,Shaniqua,Jasmine,Tanisha,Tia,Lakisha,Latoya,Yolanda,Malika,Yvette,Aisha,Ebony,Keisha,Kenya,Lakisha,Latoya,Tamika,Tanisha'.split(','))))

#Full WEAT
pleasant = ['caress','freedom','health','love','peace','cheer','friend','heaven','loyal','pleasure','diamond','gentle','honest','lucky','rainbow','diploma','gift','honor','miracle','sunrise','family','happy','laughter','paradise','vacation']
unpleasant = ['abuse','crash','filth','murder','sickness','accident','death','grief','poison','stink','assault','disaster','hatred','pollute','tragedy','divorce','jail','poverty','ugly','cancer','kill','rotten','vomit','agony','prison']
flower = ['aster','clover','hyacinth','marigold','poppy','azalea','crocus','iris','orchid','rose','bluebell','daffodil','lilac','pansy','tulip','buttercup','daisy','lily','peony','violet','carnation','gladiola','magnolia','petunia','zinnia']
insect = ['ant','caterpillar','flea','locust','spider','bedbug','centipede','fly','maggot','tarantula','bee','cockroach','gnat','mosquito','termite','beetle','cricket','hornet','moth','wasp','blackfly','dragonfly','horsefly','roach','weevil']
instrument = ['bagpipe','cello','guitar','lute','trombone','banjo','clarinet','harmonica','mandolin','trumpet','bassoon','drum','harp','oboe','tuba','bell','fiddle','harpsichord','piano','viola','bongo','flute','horn','saxophone','violin']
weapon = ['arrow','club','gun','missile','spear','axe','dagger','harpoon','pistol','sword','blade','dynamite','hatchet','rifle','tank','bomb','firearm','knife','shotgun','teargas','cannon','grenade','mace','slingshot','whip']
ea_name = ['Adam','Harry','Josh','Roger','Alan','Frank','Justin','Ryan','Andrew','Jack','Matthew','Stephen','Brad','Greg','Paul','Jonathan','Peter','Amanda','Courtney','Heather','Melanie','Katie','Betsy','Kristin','Nancy','Stephanie','Ellen','Lauren','Colleen','Emily','Megan','Rachel'] # 17 male, 15 female
aa_name = ['Alonzo','Jamel','Theo','Alphonse','Jerome','Leroy','Torrance','Darnell','Lamar','Lionel','Tyree','Deion','Lamont','Malik','Terrence','Tyrone','Lavon','Marcellus','Wardell','Nichelle','Shereen','Ebony','Latisha','Shaniqua','Jasmine','Tanisha','Tia','Lakisha','Latoya','Yolanda','Malika','Yvette'] # 19 male and 13 female
ea_name_2 = ['Brad','Brendan','Geoffrey','Greg','Brett','Matthew','Neil','Todd','Allison','Anne','Carrie','Emily','Jill','Laurie','Meredith','Sarah'] #8 and 8 male and female
aa_name_2 = ['Darnell','Hakim','Jermaine','Kareem','Jamal','Leroy','Rasheed','Tyrone','Aisha','Ebony','Keisha','Kenya','Lakisha','Latoya','Tamika','Tanisha']
pleasant_2 = ['joy','love','peace','wonderful','pleasure','friend','laughter','happy']
unpleasant_2 = ['agony','terrible','horrible','nasty','evil','war','awful','failure']
career = ['executive','management','professional','corporation','salary','office','business','career']
domestic = ['home','parents','children','family','cousins','marriage','wedding','relatives']
male_name = ['John','Paul','Mike','Kevin','Steve','Greg','Jeff','Bill']
female_name = ['Amy','Joan','Lisa','Sarah','Diana','Kate','Ann','Donna']
male = ['male','man','boy','brother','he','him','his','son']
female = ['female','woman','girl','sister','she','her','hers','daughter']
mathematics = ['math','algebra','geometry','calculus','equations','computation','numbers','addition']
art = ['poetry','art','dance','literature','novel','symphony','drama','sculpture']
male_2 = ['brother','father','uncle','grandfather','son','he','his','him']
female_2 = ['sister','mother','aunt','grandmother','daughter','she','hers','her']
science = ['science','technology','physics','chemistry','Einstein','NASA','experiment','astronomy']
art_2 = ['poetry','art','Shakespeare','dance','literature','novel','symphony','drama']
temporary = ['impermanent','unstable','variable','fleeting','short-term','brief','occasional']
permanent = ['stable','always','constant','persistent','chronic','prolonged','forever']
mental = ['sad','hopeless','gloomy','tearful','miserable','depressed']
physical = ['sick','illness','influenza','disease','virus','cancer']
young = ['Tiffany','Michelle','Cindy','Kristy','Brad','Eric','Joey','Billy']
old = ['Ethel','Bernice','Gertrude','Agnes','Cecil','Wilbert','Mortimer','Edgar']
african_american_female = ['Aisha', 'Keisha', 'Lakisha', 'Latisha', 'Latoya', 'Malika', 'Nichelle', 'Shereen', 'Tamika', 'Tanisha', 'Yolanda', 'Yvette']
european_american_male = ['Andrew', 'Brad', 'Frank', 'Geoffrey', 'Jack', 'Jonathan', 'Josh', 'Matthew', 'Neil', 'Peter', 'Roger', 'Stephen']

#overlap_ea_and_em = ['Josh', 'Roger', 'Frank', 'Andrew', 'Jack', 'Matthew', 'Stephen', 'Brad', 'Jonathan', 'Peter']

#CEAT11
intersectional_aaf = ['aggressive', 'athletic', 'bigbutt', 'confident', 'darkskinned', 'friedchicken', 'ghetto', 'loud', 'overweight', 'promiscuous', 'unfeminine', 'unintelligent', 'unrefined']
#CEAT12
emergent_intersectional_aaf = ['aggressive', 'bigbutt', 'confident', 'darkskinned', 'friedchicken', 'overweight', 'promiscuous', 'unfeminine']
#CEAT11
intersectional_eam = ['allAmerican', 'arrogant', 'attractive', 'blond', 'high-status', 'intelligent', 'leader', 'privileged', 'racist', 'rich', 'sexist', 'successful', 'tall']
#CEAT12
emergent_intersectional_eam_2 = ['arrogant', 'blond', 'high-status', 'intelligent', 'racist', 'rich', 'successful', 'tall']
#C13
mexican_american_females = ['Adriana', 'Alejandra', 'Alma', 'Brenda', 'Carolina', 'Iliana', 'Karina', 'Liset', 'Maria', 'Mayra', 'Sonia', 'Yesenia']
intersectional_maf = ['cook', 'curvy', 'darkskinned', 'feisty', 'hardworker', 'loud', 'maids', 'promiscuous', 'sexy', 'short', 'uneducated', 'unintelligent']
#C13 (different to #C11)
emergent_intersectional_eam_3 = ['allAmerican', 'arrogant', 'blond', 'high-status', 'intelligent', 'leader', 'privileged', 'racist', 'rich', 'sexist', 'successful', 'tall']
#C14
emergent_intersec_maf = ['cook', 'curvy', 'feisty', 'maids', 'promiscuous', 'sexy']
#C14:
emergent_intersec_eam = ['arrogant', 'assertive', 'intelligent', 'rich', 'successful', 'tall']

#VAST | Greedily obtained from ANEW lexicon - multiply tokenized by GPT-2
multi_pleasant = ['masterful','dignified','politeness','easygoing','sailboat','blossom','dazzle','soothe','fascinate','jolly','refreshment','elated','luscious','carefree','untroubled','cuddle','christmas','caress','snuggle','rollercoaster','valentine','sweetheart']
multi_unpleasant = ['suffocate','syphilis','rabies','ulcer','mutilate','pollute','morgue','disloyal','toothache','seasick','unfaithful','maggot','leprosy','anguished','detest','stench','crucify','humiliate','gangrene','regretful','lice','roach','putrid']

multi_dominance = ['bathtub','glamour','carefree','nourish','valentine','garter','lightbulb','caress','detest','cuddle','sailboat','swimmer','zest','sweetheart','decorate','dignified','bouquet','fascinate','jolly','penthouse','masterful']
multi_submission = ['humiliate','shamed','unfaithful','flabby','syphilis','gangrene','mutilate','seasick','despairing','impotent','regretful','suffocate','anguished','scapegoat','ache','louse','sissy','morgue','meek','crucify','wasp','deserter']

multi_arousal = ['valentine','pollute','rabies','cockroach','ulcer','humiliate','unfaithful','elated','pervert','christmas','leprosy','dazzle','cyclone','mutilate','crucify','disloyal','guillotine','roach','infatuation','skijump','rollercoaster']
multi_indifferent = ['fatigued','dreary','nonchalant','hairpin','mantel','mucus','prairie','dustpan','kerchief','overcast','utensil','hairdryer','hydrant','golfer','slush','politeness','windmill','thermometer','cork','leisurely','meek','handicap']



#Scripting Area

weat_terms_weat = list(set(flower + insect + instrument + weapon + ea_name + aa_name + ea_name_2 + aa_name_2 + pleasant + unpleasant + pleasant_2 + unpleasant_2 + young + old + male_name + female_name + career + domestic + male + female + science + mathematics + art + art_2))
pleasant_weat = list(set(flower + instrument + ea_name + ea_name_2 + pleasant + pleasant_2 + young))
unpleasant_weat = list(set(insect + weapon + aa_name + aa_name_2 + unpleasant + unpleasant_2 + old))
neutral_weat = list(set(male_name + female_name + career + domestic + male + female + science + mathematics + art + art_2))

#VAST
weat_terms_vast = list(set(weat_terms_weat + dominant + arousal + submissive + indifference))




