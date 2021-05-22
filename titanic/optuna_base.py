import optuna

def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)

study.best_params  # E.g. {'x': 2.002108042}
'''
[I 2021-05-22 21:50:25,855] Trial 0 finished with value: 4.901344506787557 and parameters: {'x': -0.21389803441521593}. Best is trial 0 with value: 4.901344506787557.
[I 2021-05-22 21:50:25,856] Trial 1 finished with value: 19.383686822446574 and parameters: {'x': 6.402690861558028}. Best is trial 0 with value: 4.901344506787557.
[I 2021-05-22 21:50:25,857] Trial 2 finished with value: 48.26980998276973 and parameters: {'x': -4.947647802153599}. Best is trial 0 with value: 4.901344506787557.
[I 2021-05-22 21:50:25,857] Trial 3 finished with value: 2.2924231332863716 and parameters: {'x': 3.514075009134743}. Best is trial 3 with value: 2.2924231332863716.
[I 2021-05-22 21:50:25,858] Trial 4 finished with value: 50.07509975290213 and parameters: {'x': 9.0763761737843}. Best is trial 3 with value: 2.2924231332863716.
[I 2021-05-22 21:50:25,858] Trial 5 finished with value: 36.42002000004158 and parameters: {'x': -4.034900164877757}. Best is trial 3 with value: 2.2924231332863716.
[I 2021-05-22 21:50:25,859] Trial 6 finished with value: 43.68813229659039 and parameters: {'x': 8.60969986433502}. Best 
is trial 3 with value: 2.2924231332863716.
[I 2021-05-22 21:50:25,859] Trial 7 finished with value: 88.32044291040285 and parameters: {'x': -7.397895663945353}. Best is trial 3 with value: 2.2924231332863716.
[I 2021-05-22 21:50:25,860] Trial 8 finished with value: 1.7977545732486462 and parameters: {'x': 0.6591962957805322}. Best is trial 8 with value: 1.7977545732486462.
[I 2021-05-22 21:50:25,860] Trial 9 finished with value: 72.69386502381275 and parameters: {'x': -6.526069729002499}. Best is trial 8 with value: 1.7977545732486462.
[I 2021-05-22 21:50:25,863] Trial 10 finished with value: 3.5739680535044713 and parameters: {'x': 0.10950587054482741}. 
Best is trial 8 with value: 1.7977545732486462.
[I 2021-05-22 21:50:25,866] Trial 11 finished with value: 2.176204566624628 and parameters: {'x': 3.4751964501803236}. Best is trial 8 with value: 1.7977545732486462.
[I 2021-05-22 21:50:25,868] Trial 12 finished with value: 0.1306044295879967 and parameters: {'x': 2.3613923485465578}. Best is trial 12 with value: 0.1306044295879967.
[I 2021-05-22 21:50:25,871] Trial 13 finished with value: 7.199607068301887 and parameters: {'x': -0.6832083535018085}. Best is trial 12 with value: 0.1306044295879967.
[I 2021-05-22 21:50:25,873] Trial 14 finished with value: 1.2378109993631194 and parameters: {'x': 3.1125695481016544}. Best is trial 12 with value: 0.1306044295879967.
[I 2021-05-22 21:50:25,875] Trial 15 finished with value: 5.881056497906847 and parameters: {'x': 4.425088967008602}. Best is trial 12 with value: 0.1306044295879967.
[I 2021-05-22 21:50:25,878] Trial 16 finished with value: 19.991225538861613 and parameters: {'x': 6.471154832798973}. Best is trial 12 with value: 0.1306044295879967.
[I 2021-05-22 21:50:25,880] Trial 17 finished with value: 18.097579313552274 and parameters: {'x': -2.2541249762497895}. 
Best is trial 12 with value: 0.1306044295879967.
[I 2021-05-22 21:50:25,883] Trial 18 finished with value: 0.04006260563388987 and parameters: {'x': 1.799843547109043}. Best is trial 18 with value: 0.04006260563388987.
[I 2021-05-22 21:50:25,885] Trial 19 finished with value: 0.002413113233474782 and parameters: {'x': 1.950876551083268}. 
Best is trial 19 with value: 0.002413113233474782.
[I 2021-05-22 21:50:25,887] Trial 20 finished with value: 129.71755082662412 and parameters: {'x': -9.389361300205737}. Best is trial 19 with value: 0.002413113233474782.
[I 2021-05-22 21:50:25,889] Trial 21 finished with value: 0.36320688208250457 and parameters: {'x': 1.3973335233460347}. 
Best is trial 19 with value: 0.002413113233474782.
[I 2021-05-22 21:50:25,892] Trial 22 finished with value: 0.0001617458402431121 and parameters: {'x': 2.0127179338040073}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,894] Trial 23 finished with value: 11.698491062677439 and parameters: {'x': 5.420305697255355}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,897] Trial 24 finished with value: 0.07018230402865076 and parameters: {'x': 1.7350805706848764}. 
Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,898] Trial 25 finished with value: 17.491757675785756 and parameters: {'x': -2.182314870473738}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,901] Trial 26 finished with value: 15.26479831173421 and parameters: {'x': -1.9070191081864714}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,903] Trial 27 finished with value: 9.044742820286011 and parameters: {'x': 5.007447891532954}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,905] Trial 28 finished with value: 25.263666536776572 and parameters: {'x': 7.026297497838401}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,907] Trial 29 finished with value: 7.808251962566109 and parameters: {'x': -0.7943249565084782}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,910] Trial 30 finished with value: 0.24693027241980078 and parameters: {'x': 1.5030792091089358}. 
Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,912] Trial 31 finished with value: 0.0011711610728748134 and parameters: {'x': 2.034222230682333}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,915] Trial 32 finished with value: 3.332405778693436 and parameters: {'x': 0.17451218062309826}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,917] Trial 33 finished with value: 0.5660809486806538 and parameters: {'x': 2.7523835117017477}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,919] Trial 34 finished with value: 2.7035296107860853 and parameters: {'x': 3.64424134809525}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,922] Trial 35 finished with value: 10.465586799452828 and parameters: {'x': -1.2350559190611883}. 
Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,924] Trial 36 finished with value: 26.26126059493959 and parameters: {'x': -3.1245741866948893}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,926] Trial 37 finished with value: 1.7630132117395894 and parameters: {'x': 0.672214922609992}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,929] Trial 38 finished with value: 27.91389362452066 and parameters: {'x': 7.2833600695505}. Best 
is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,931] Trial 39 finished with value: 10.205855852892416 and parameters: {'x': 5.194660522323525}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,933] Trial 40 finished with value: 0.00790089577393527 and parameters: {'x': 2.0888869831524013}. 
Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,936] Trial 41 finished with value: 0.014543613602186093 and parameters: {'x': 1.8794030945580025}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,938] Trial 42 finished with value: 4.968386155799115 and parameters: {'x': 4.22898769754324}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,941] Trial 43 finished with value: 1.1709186516033425 and parameters: {'x': 0.9179100538294691}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,943] Trial 44 finished with value: 0.1586277915481759 and parameters: {'x': 2.3982810459313573}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,946] Trial 45 finished with value: 4.217451858426071 and parameters: {'x': -0.05364355680971844}. 
Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,948] Trial 46 finished with value: 3.9752722642925296 and parameters: {'x': 3.993808482350431}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,950] Trial 47 finished with value: 0.47946214438468754 and parameters: {'x': 2.692432050373672}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,953] Trial 48 finished with value: 62.483644430320595 and parameters: {'x': 9.904659665685841}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,955] Trial 49 finished with value: 0.048662094979580725 and parameters: {'x': 2.22059486616778}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,958] Trial 50 finished with value: 15.71250506922573 and parameters: {'x': 5.96390023451975}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,960] Trial 51 finished with value: 0.1412136480752915 and parameters: {'x': 1.624215955533911}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,962] Trial 52 finished with value: 1.5510223433985533 and parameters: {'x': 3.2454004751077274}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,965] Trial 53 finished with value: 1.9179819904702584 and parameters: {'x': 0.615087731850765}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,967] Trial 54 finished with value: 0.00394903813912439 and parameters: {'x': 2.062841372829724}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,969] Trial 55 finished with value: 6.563443187198039 and parameters: {'x': 4.5619217761668756}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,972] Trial 56 finished with value: 6.694270416645001 and parameters: {'x': -0.5873288188100483}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,974] Trial 57 finished with value: 1.6676276174999647 and parameters: {'x': 3.291366569762422}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,977] Trial 58 finished with value: 0.9770354571724836 and parameters: {'x': 1.0115489606599204}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,979] Trial 59 finished with value: 0.016692722590315293 and parameters: {'x': 2.129200319621568}. 
Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,981] Trial 60 finished with value: 11.513773106168442 and parameters: {'x': -1.3931951176094253}. 
Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,984] Trial 61 finished with value: 0.023980705329027064 and parameters: {'x': 2.1548570480444047}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,986] Trial 62 finished with value: 3.1962993563617643 and parameters: {'x': 0.21218027856224708}. 
Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,988] Trial 63 finished with value: 0.6488762683307745 and parameters: {'x': 1.194471435434612}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,991] Trial 64 finished with value: 0.7282388478201142 and parameters: {'x': 2.853369115811039}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,993] Trial 65 finished with value: 3.73878038753319 and parameters: {'x': 3.933592611573904}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,996] Trial 66 finished with value: 0.06990755605526838 and parameters: {'x': 1.7355996292452138}. 
Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:25,998] Trial 67 finished with value: 0.03761415250331959 and parameters: {'x': 2.193943683844872}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,000] Trial 68 finished with value: 5.348792873496344 and parameters: {'x': -0.31274574337438654}. 
Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,003] Trial 69 finished with value: 0.4303308101232632 and parameters: {'x': 2.655996044289341}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,005] Trial 70 finished with value: 2.2308570404049406 and parameters: {'x': 3.4936053830931852}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,007] Trial 71 finished with value: 0.040125996917230386 and parameters: {'x': 2.2003147446326166}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,010] Trial 72 finished with value: 1.9213920826047528 and parameters: {'x': 0.6138571204220133}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,012] Trial 73 finished with value: 7.643759778537462 and parameters: {'x': 4.7647350286306756}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,015] Trial 74 finished with value: 0.07435547083117378 and parameters: {'x': 1.7273180042042127}. 
Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,017] Trial 75 finished with value: 0.7710655593570136 and parameters: {'x': 1.121896612375847}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,020] Trial 76 finished with value: 1.1550497603338397 and parameters: {'x': 3.0747324133633636}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,022] Trial 77 finished with value: 0.006414340887755499 and parameters: {'x': 2.0800895803944277}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,024] Trial 78 finished with value: 3.776358434030563 and parameters: {'x': 0.05671452585304304}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,027] Trial 79 finished with value: 3.2283971553820603 and parameters: {'x': 3.7967740969253927}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,029] Trial 80 finished with value: 0.23347019735632835 and parameters: {'x': 1.5168124615055472}. 
Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,032] Trial 81 finished with value: 0.060693068976982796 and parameters: {'x': 2.2463596334162372}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,034] Trial 82 finished with value: 0.007116091722496497 and parameters: {'x': 2.0843569304947525}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,036] Trial 83 finished with value: 2.0511049201272207 and parameters: {'x': 0.5678320908052643}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,039] Trial 84 finished with value: 0.9054159092305739 and parameters: {'x': 2.951533451451169}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,041] Trial 85 finished with value: 0.5333191385858052 and parameters: {'x': 1.2697129752037182}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,044] Trial 86 finished with value: 13.178958517758396 and parameters: {'x': 5.63028353131796}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,046] Trial 87 finished with value: 4.886253538491983 and parameters: {'x': 4.210487172207064}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,049] Trial 88 finished with value: 0.31346156100077577 and parameters: {'x': 2.5598763801061586}. 
Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,051] Trial 89 finished with value: 0.0019769126160889667 and parameters: {'x': 1.95553751450842}. 
Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,053] Trial 90 finished with value: 10.976681221273148 and parameters: {'x': -1.3131074871294395}. 
Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,056] Trial 91 finished with value: 0.008814719543113445 and parameters: {'x': 1.9061132621553318}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,059] Trial 92 finished with value: 0.04685777035428319 and parameters: {'x': 1.7835334428733085}. 
Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,062] Trial 93 finished with value: 2.0720458628292877 and parameters: {'x': 3.4394602678883803}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,064] Trial 94 finished with value: 2.4859931571708285 and parameters: {'x': 0.42329674409836127}. 
Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,067] Trial 95 finished with value: 1.0808769981917132 and parameters: {'x': 0.9603476551309504}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,069] Trial 96 finished with value: 0.8810874379171446 and parameters: {'x': 2.93866257937405}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,072] Trial 97 finished with value: 5.522709236341806 and parameters: {'x': -0.3500445179489273}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,075] Trial 98 finished with value: 0.3600695597435451 and parameters: {'x': 1.3999420363468666}. Best is trial 22 with value: 0.0001617458402431121.
[I 2021-05-22 21:50:26,077] Trial 99 finished with value: 0.003693533347135864 and parameters: {'x': 2.0607744464979803}. Best is trial 22 with value: 0.0001617458402431121.
'''