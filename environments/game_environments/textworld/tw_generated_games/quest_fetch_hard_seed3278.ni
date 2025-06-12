Use MAX_STATIC_DATA of 500000.
When play begins, seed the random-number generator with 1234.

container is a kind of thing.
door is a kind of thing.
object-like is a kind of thing.
supporter is a kind of thing.
food is a kind of object-like.
key is a kind of object-like.
containers are openable, lockable and fixed in place. containers are usually closed.
door is openable and lockable.
object-like is portable.
supporters are fixed in place.
food is edible.
A room has a text called internal name.


The r_2 and the r_3 and the r_4 and the r_8 and the r_5 and the r_6 and the r_0 and the r_1 and the r_7 and the r_9 are rooms.

Understand "balmy canteen" as r_2.
The internal name of r_2 is "balmy canteen".
The printed name of r_2 is "-= Balmy Canteen =-".
The balmy canteen part 0 is some text that varies. The balmy canteen part 0 is "You arrive in a canteen. A balmy one.



 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The balmy canteen part 1 is some text that varies. The balmy canteen part 1 is " birch non-euclidean hatch leading north. There is [if d_0 is open]an open[otherwise]a closed[end if]".
The balmy canteen part 2 is some text that varies. The balmy canteen part 2 is " wooden gateway leading west. There is an unblocked exit to the south.".
The description of r_2 is "[balmy canteen part 0][balmy canteen part 1][balmy canteen part 2]".

west of r_2 and east of r_3 is a door called d_0.
The r_9 is mapped south of r_2.
north of r_2 and south of r_0 is a door called d_1.
Understand "cozy chamber" as r_3.
The internal name of r_3 is "cozy chamber".
The printed name of r_3 is "-= Cozy Chamber =-".
The cozy chamber part 0 is some text that varies. The cozy chamber part 0 is "You find yourself in a chamber. A cozy kind of place.

 You see a worn-out bed. [if there is something on the s_0]You see [a list of things on the s_0] on the bed.[end if]".
The cozy chamber part 1 is some text that varies. The cozy chamber part 1 is "[if there is nothing on the s_0]But oh no! there's nothing on this piece of garbage.[end if]".
The cozy chamber part 2 is some text that varies. The cozy chamber part 2 is "

 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The cozy chamber part 3 is some text that varies. The cozy chamber part 3 is " wooden gateway leading east. There is an exit to the north. Don't worry, it is unblocked. You don't like doors? Why not try going south, that entranceway is unblocked.".
The description of r_3 is "[cozy chamber part 0][cozy chamber part 1][cozy chamber part 2][cozy chamber part 3]".

The r_4 is mapped south of r_3.
The r_5 is mapped north of r_3.
east of r_3 and west of r_2 is a door called d_0.
Understand "spacious garage" as r_4.
The internal name of r_4 is "spacious garage".
The printed name of r_4 is "-= Spacious Garage =-".
The spacious garage part 0 is some text that varies. The spacious garage part 0 is "You find yourself in a spacious garage.



There is an unblocked exit to the north. You need an unguarded exit? You should try going west.".
The description of r_4 is "[spacious garage part 0]".

The r_8 is mapped west of r_4.
The r_3 is mapped north of r_4.
Understand "steamy canteen" as r_8.
The internal name of r_8 is "steamy canteen".
The printed name of r_8 is "-= Steamy Canteen =-".
The steamy canteen part 0 is some text that varies. The steamy canteen part 0 is "You arrive in a canteen. A steamy kind of place.

 [if c_1 is locked]A locked[else if c_1 is open]An open[otherwise]A closed[end if]".
The steamy canteen part 1 is some text that varies. The steamy canteen part 1 is " chest, which looks fancy, is in the room.[if c_1 is open and there is something in the c_1] The fancy chest contains [a list of things in the c_1]. Is this it? Is this what you came to TextWorld to see? a chest?[end if]".
The steamy canteen part 2 is some text that varies. The steamy canteen part 2 is "[if c_1 is open and the c_1 contains nothing] The chest is empty, what a horrible day![end if]".
The steamy canteen part 3 is some text that varies. The steamy canteen part 3 is "

There is an exit to the east. Don't worry, it is unblocked. You don't like doors? Why not try going north, that entranceway is unguarded.".
The description of r_8 is "[steamy canteen part 0][steamy canteen part 1][steamy canteen part 2][steamy canteen part 3]".

The r_7 is mapped north of r_8.
The r_4 is mapped east of r_8.
Understand "clean restroom" as r_5.
The internal name of r_5 is "clean restroom".
The printed name of r_5 is "-= Clean Restroom =-".
The clean restroom part 0 is some text that varies. The clean restroom part 0 is "You arrive in a clean restroom.



There is an unguarded exit to the south. You need an unguarded exit? You should try going west.".
The description of r_5 is "[clean restroom part 0]".

The r_6 is mapped west of r_5.
The r_3 is mapped south of r_5.
Understand "entertaining lounge" as r_6.
The internal name of r_6 is "entertaining lounge".
The printed name of r_6 is "-= Entertaining Lounge =-".
The entertaining lounge part 0 is some text that varies. The entertaining lounge part 0 is "You make a grand eccentric entrance into an entertaining lounge.

 You can see [if c_2 is locked]a locked[else if c_2 is open]an opened[otherwise]a closed[end if]".
The entertaining lounge part 1 is some text that varies. The entertaining lounge part 1 is " portmanteau.[if c_2 is open and there is something in the c_2] The dusty portmanteau contains [a list of things in the c_2].[end if]".
The entertaining lounge part 2 is some text that varies. The entertaining lounge part 2 is "[if c_2 is open and the c_2 contains nothing] The portmanteau is empty! This is the worst thing that could possibly happen, ever![end if]".
The entertaining lounge part 3 is some text that varies. The entertaining lounge part 3 is " Were you looking for a safe? Because look over there, it's a safe.[if c_3 is open and there is something in the c_3] The new safe contains [a list of things in the c_3].[end if]".
The entertaining lounge part 4 is some text that varies. The entertaining lounge part 4 is "[if c_3 is open and the c_3 contains nothing] The safe is empty! This is the worst thing that could possibly happen, ever![end if]".
The entertaining lounge part 5 is some text that varies. The entertaining lounge part 5 is "

You don't like doors? Why not try going east, that entranceway is unguarded. There is an exit to the south. Don't worry, it is unblocked.".
The description of r_6 is "[entertaining lounge part 0][entertaining lounge part 1][entertaining lounge part 2][entertaining lounge part 3][entertaining lounge part 4][entertaining lounge part 5]".

The r_7 is mapped south of r_6.
The r_5 is mapped east of r_6.
Understand "clean washroom" as r_0.
The internal name of r_0 is "clean washroom".
The printed name of r_0 is "-= Clean Washroom =-".
The clean washroom part 0 is some text that varies. The clean washroom part 0 is "You've just sauntered into a clean washroom.

 You lean against the wall, inadvertently pressing a secret button. The wall opens up to reveal a chipped rack. [if there is something on the s_1]On the chipped rack you can make out [a list of things on the s_1].[end if]".
The clean washroom part 1 is some text that varies. The clean washroom part 1 is "[if there is nothing on the s_1]Looks like someone's already been here and taken everything off it, though.[end if]".
The clean washroom part 2 is some text that varies. The clean washroom part 2 is "

 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The clean washroom part 3 is some text that varies. The clean washroom part 3 is " birch non-euclidean hatch leading south. There is an exit to the north. Don't worry, it is unblocked.".
The description of r_0 is "[clean washroom part 0][clean washroom part 1][clean washroom part 2][clean washroom part 3]".

south of r_0 and north of r_2 is a door called d_1.
The r_1 is mapped north of r_0.
Understand "gloomy basement" as r_1.
The internal name of r_1 is "gloomy basement".
The printed name of r_1 is "-= Gloomy Basement =-".
The gloomy basement part 0 is some text that varies. The gloomy basement part 0 is "You find yourself in a basement. A gloomy one.

 You can see a brand new case.[if c_0 is open and there is something in the c_0] The brand new case contains [a list of things in the c_0].[end if]".
The gloomy basement part 1 is some text that varies. The gloomy basement part 1 is "[if c_0 is open and the c_0 contains nothing] The case is empty! This is the worst thing that could possibly happen, ever![end if]".
The gloomy basement part 2 is some text that varies. The gloomy basement part 2 is "

You need an unblocked exit? You should try going south.".
The description of r_1 is "[gloomy basement part 0][gloomy basement part 1][gloomy basement part 2]".

The r_0 is mapped south of r_1.
Understand "ugly pantry" as r_7.
The internal name of r_7 is "ugly pantry".
The printed name of r_7 is "-= Ugly Pantry =-".
The ugly pantry part 0 is some text that varies. The ugly pantry part 0 is "You find yourself in a pantry. An ugly kind of place.

 You make out [if c_4 is locked]a locked[else if c_4 is open]an opened[otherwise]a closed[end if]".
The ugly pantry part 1 is some text that varies. The ugly pantry part 1 is " sturdy type K chest.[if c_4 is open and there is something in the c_4] The sturdy type K chest contains [a list of things in the c_4].[end if]".
The ugly pantry part 2 is some text that varies. The ugly pantry part 2 is "[if c_4 is open and the c_4 contains nothing] The type K chest is empty! This is the worst thing that could possibly happen, ever![end if]".
The ugly pantry part 3 is some text that varies. The ugly pantry part 3 is "

There is an unguarded exit to the north. There is an unblocked exit to the south.".
The description of r_7 is "[ugly pantry part 0][ugly pantry part 1][ugly pantry part 2][ugly pantry part 3]".

The r_8 is mapped south of r_7.
The r_6 is mapped north of r_7.
Understand "spotless steam room" as r_9.
The internal name of r_9 is "spotless steam room".
The printed name of r_9 is "-= Spotless Steam Room =-".
The spotless steam room part 0 is some text that varies. The spotless steam room part 0 is "You have come into a steam room. Not the steam room you'd expect. No, this is a spotless steam room. You can barely contain your excitement.

 You can make out [if c_5 is locked]a locked[else if c_5 is open]an opened[otherwise]a closed[end if]".
The spotless steam room part 1 is some text that varies. The spotless steam room part 1 is " cabinet in the room.[if c_5 is open and there is something in the c_5] The cedarwood cabinet contains [a list of things in the c_5]. You shudder, but continue examining the room.[end if]".
The spotless steam room part 2 is some text that varies. The spotless steam room part 2 is "[if c_5 is open and the c_5 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The spotless steam room part 3 is some text that varies. The spotless steam room part 3 is " You see a counter. The counter is chipped.[if there is something on the s_2] On the chipped counter you see [a list of things on the s_2]. There's something strange about this thing being here, but you don't have time to worry about that now.[end if]".
The spotless steam room part 4 is some text that varies. The spotless steam room part 4 is "[if there is nothing on the s_2] But the thing is empty, unfortunately.[end if]".
The spotless steam room part 5 is some text that varies. The spotless steam room part 5 is "

There is an unblocked exit to the north.".
The description of r_9 is "[spotless steam room part 0][spotless steam room part 1][spotless steam room part 2][spotless steam room part 3][spotless steam room part 4][spotless steam room part 5]".

The r_2 is mapped north of r_9.

The c_0 and the c_1 and the c_2 and the c_3 and the c_4 and the c_5 are containers.
The c_0 and the c_1 and the c_2 and the c_3 and the c_4 and the c_5 are privately-named.
The d_1 and the d_0 are doors.
The d_1 and the d_0 are privately-named.
The f_1 and the f_0 are foods.
The f_1 and the f_0 are privately-named.
The k_0 and the k_1 are keys.
The k_0 and the k_1 are privately-named.
The o_0 and the o_1 are object-likes.
The o_0 and the o_1 are privately-named.
The r_2 and the r_3 and the r_4 and the r_8 and the r_5 and the r_6 and the r_0 and the r_1 and the r_7 and the r_9 are rooms.
The r_2 and the r_3 and the r_4 and the r_8 and the r_5 and the r_6 and the r_0 and the r_1 and the r_7 and the r_9 are privately-named.
The s_0 and the s_1 and the s_2 are supporters.
The s_0 and the s_1 and the s_2 are privately-named.

The description of d_1 is "it's a stuffy non-euclidean hatch [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_1 is "birch non-euclidean hatch".
Understand "birch non-euclidean hatch" as d_1.
Understand "birch" as d_1.
Understand "non-euclidean" as d_1.
Understand "hatch" as d_1.
The d_1 is open.
The description of d_0 is "it's a noble gateway [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_0 is "wooden gateway".
Understand "wooden gateway" as d_0.
Understand "wooden" as d_0.
Understand "gateway" as d_0.
The d_0 is open.
The description of c_0 is "The brand new case looks strong, and impossible to destroy. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_0 is "brand new case".
Understand "brand new case" as c_0.
Understand "brand" as c_0.
Understand "new" as c_0.
Understand "case" as c_0.
The c_0 is in r_1.
The c_0 is open.
The description of c_1 is "The fancy chest looks strong, and impossible to destroy. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_1 is "fancy chest".
Understand "fancy chest" as c_1.
Understand "fancy" as c_1.
Understand "chest" as c_1.
The c_1 is in r_8.
The c_1 is open.
The description of c_2 is "The dusty portmanteau looks strong, and impossible to break. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_2 is "dusty portmanteau".
Understand "dusty portmanteau" as c_2.
Understand "dusty" as c_2.
Understand "portmanteau" as c_2.
The c_2 is in r_6.
The c_2 is open.
The description of c_3 is "The new safe looks strong, and impossible to destroy. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_3 is "new safe".
Understand "new safe" as c_3.
Understand "new" as c_3.
Understand "safe" as c_3.
The c_3 is in r_6.
The c_3 is open.
The description of c_4 is "The sturdy type K chest looks strong, and impossible to destroy. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_4 is "sturdy type K chest".
Understand "sturdy type K chest" as c_4.
Understand "sturdy" as c_4.
Understand "type" as c_4.
Understand "K" as c_4.
Understand "chest" as c_4.
The c_4 is in r_7.
The c_4 is closed.
The description of c_5 is "The cedarwood cabinet looks strong, and impossible to break. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_5 is "cedarwood cabinet".
Understand "cedarwood cabinet" as c_5.
Understand "cedarwood" as c_5.
Understand "cabinet" as c_5.
The c_5 is in r_9.
The c_5 is open.
The description of f_1 is "The half-eaten stick of butter looks tasty.".
The printed name of f_1 is "half-eaten stick of butter".
Understand "half-eaten stick of butter" as f_1.
Understand "half-eaten" as f_1.
Understand "stick" as f_1.
Understand "butter" as f_1.
The f_1 is in r_5.
The f_1 is edible.
The description of s_0 is "The worn-out bed is durable.".
The printed name of s_0 is "worn-out bed".
Understand "worn-out bed" as s_0.
Understand "worn-out" as s_0.
Understand "bed" as s_0.
The s_0 is in r_3.
The description of s_1 is "The chipped rack is shaky.".
The printed name of s_1 is "chipped rack".
Understand "chipped rack" as s_1.
Understand "chipped" as s_1.
Understand "rack" as s_1.
The s_1 is in r_0.
The description of s_2 is "The chipped counter is solid.".
The printed name of s_2 is "chipped counter".
Understand "chipped counter" as s_2.
Understand "chipped" as s_2.
Understand "counter" as s_2.
The s_2 is in r_9.
The description of f_0 is "that's a half-eaten cashew!".
The printed name of f_0 is "half-eaten cashew".
Understand "half-eaten cashew" as f_0.
Understand "half-eaten" as f_0.
Understand "cashew" as f_0.
The f_0 is edible.
The player carries the f_0.
The description of k_0 is "The iron non-euclidean latchkey is cold to the touch".
The printed name of k_0 is "iron non-euclidean latchkey".
Understand "iron non-euclidean latchkey" as k_0.
Understand "iron" as k_0.
Understand "non-euclidean" as k_0.
Understand "latchkey" as k_0.
The k_0 is in the c_0.
The matching key of the d_1 is the k_0.
The description of k_1 is "The metal type K passkey looks useful".
The printed name of k_1 is "metal type K passkey".
Understand "metal type K passkey" as k_1.
Understand "metal" as k_1.
Understand "type" as k_1.
Understand "K" as k_1.
Understand "passkey" as k_1.
The player carries the k_1.
The matching key of the c_4 is the k_1.
The description of o_0 is "The austere fork is dirty.".
The printed name of o_0 is "austere fork".
Understand "austere fork" as o_0.
Understand "austere" as o_0.
Understand "fork" as o_0.
The player carries the o_0.
The description of o_1 is "The tacky scarf would seem to be out of place here".
The printed name of o_1 is "tacky scarf".
Understand "tacky scarf" as o_1.
Understand "tacky" as o_1.
Understand "scarf" as o_1.
The player carries the o_1.


The player is in r_0.

The quest0 completed is a truth state that varies.
The quest0 completed is usually false.

Test quest0_0 with "close birch non-euclidean hatch"

Every turn:
	if quest0 completed is true:
		do nothing;
	else if The player is in r_0 and The d_1 is closed and the d_1 is unlocked:
		increase the score by 1; [Quest completed]
		if 1 is 1 [always true]:
			Now the quest0 completed is true;

The quest1 completed is a truth state that varies.
The quest1 completed is usually false.

Test quest1_0 with "close birch non-euclidean hatch / go north / take iron non-euclidean latchkey from brand new case"

Every turn:
	if quest1 completed is true:
		do nothing;
	else if The player is in r_1 and The c_0 is in r_1 and The c_0 is open and The player carries the k_0:
		increase the score by 1; [Quest completed]
		if 1 is 1 [always true]:
			Now the quest1 completed is true;

The quest2 completed is a truth state that varies.
The quest2 completed is usually false.

Test quest2_0 with "close birch non-euclidean hatch / go north / take iron non-euclidean latchkey from brand new case / go south / open birch non-euclidean hatch / go south / close birch non-euclidean hatch / lock birch non-euclidean hatch with iron non-euclidean latchkey"

Every turn:
	if quest2 completed is true:
		do nothing;
	else if The player is in r_2 and The player carries the k_0 and The matching key of the d_1 is the k_0 and The d_1 is locked:
		increase the score by 1; [Quest completed]
		if 1 is 1 [always true]:
			Now the quest2 completed is true;

Use scoring. The maximum score is 3.
This is the simpler notify score changes rule:
	If the score is not the last notified score:
		let V be the score - the last notified score;
		if V > 0:
			say "Your score has just gone up by [V in words] ";
		else:
			say "Your score changed by [V in words] ";
		if V >= -1 and V <= 1:
			say "point.";
		else:
			say "points.";
		Now the last notified score is the score;
	if quest0 completed is true and quest1 completed is true and quest2 completed is true:
		end the story finally; [Win]

The simpler notify score changes rule substitutes for the notify score changes rule.

Rule for listing nondescript items:
	stop.

Rule for printing the banner text:
	say "[fixed letter spacing]";
	say "                    ________  ________  __    __  ________        [line break]";
	say "                   |        \|        \|  \  |  \|        \       [line break]";
	say "                    \$$$$$$$$| $$$$$$$$| $$  | $$ \$$$$$$$$       [line break]";
	say "                      | $$   | $$__     \$$\/  $$   | $$          [line break]";
	say "                      | $$   | $$  \     >$$  $$    | $$          [line break]";
	say "                      | $$   | $$$$$    /  $$$$\    | $$          [line break]";
	say "                      | $$   | $$_____ |  $$ \$$\   | $$          [line break]";
	say "                      | $$   | $$     \| $$  | $$   | $$          [line break]";
	say "                       \$$    \$$$$$$$$ \$$   \$$    \$$          [line break]";
	say "              __       __   ______   _______   __        _______  [line break]";
	say "             |  \  _  |  \ /      \ |       \ |  \      |       \ [line break]";
	say "             | $$ / \ | $$|  $$$$$$\| $$$$$$$\| $$      | $$$$$$$\[line break]";
	say "             | $$/  $\| $$| $$  | $$| $$__| $$| $$      | $$  | $$[line break]";
	say "             | $$  $$$\ $$| $$  | $$| $$    $$| $$      | $$  | $$[line break]";
	say "             | $$ $$\$$\$$| $$  | $$| $$$$$$$\| $$      | $$  | $$[line break]";
	say "             | $$$$  \$$$$| $$__/ $$| $$  | $$| $$_____ | $$__/ $$[line break]";
	say "             | $$$    \$$$ \$$    $$| $$  | $$| $$     \| $$    $$[line break]";
	say "              \$$      \$$  \$$$$$$  \$$   \$$ \$$$$$$$$ \$$$$$$$ [line break]";
	say "[variable letter spacing][line break]";
	say "[objective][line break]".

Include Basic Screen Effects by Emily Short.

Rule for printing the player's obituary:
	if story has ended finally:
		center "*** The End ***";
	else:
		center "*** You lost! ***";
	say paragraph break;
	if maximum score is -32768:
		say "You scored a total of [score] point[s], in [turn count] turn[s].";
	else:
		say "You scored [score] out of a possible [maximum score], in [turn count] turn[s].";
	[wait for any key;
	stop game abruptly;]
	rule succeeds.

Carry out requesting the score:
	if maximum score is -32768:
		say "You have so far scored [score] point[s], in [turn count] turn[s].";
	else:
		say "You have so far scored [score] out of a possible [maximum score], in [turn count] turn[s].";
	rule succeeds.

Rule for implicitly taking something (called target):
	if target is fixed in place:
		say "The [target] is fixed in place.";
	otherwise:
		say "You need to take the [target] first.";
		set pronouns from target;
	stop.

Does the player mean doing something:
	if the noun is not nothing and the second noun is nothing and the player's command matches the text printed name of the noun:
		it is likely;
	if the noun is nothing and the second noun is not nothing and the player's command matches the text printed name of the second noun:
		it is likely;
	if the noun is not nothing and the second noun is not nothing and the player's command matches the text printed name of the noun and the player's command matches the text printed name of the second noun:
		it is very likely.  [Handle action with two arguments.]

Printing the content of the room is an activity.
Rule for printing the content of the room:
	let R be the location of the player;
	say "Room contents:[line break]";
	list the contents of R, with newlines, indented, including all contents, with extra indentation.

Printing the content of the world is an activity.
Rule for printing the content of the world:
	let L be the list of the rooms;
	say "World: [line break]";
	repeat with R running through L:
		say "  [the internal name of R][line break]";
	repeat with R running through L:
		say "[the internal name of R]:[line break]";
		if the list of things in R is empty:
			say "  nothing[line break]";
		otherwise:
			list the contents of R, with newlines, indented, including all contents, with extra indentation.

Printing the content of the inventory is an activity.
Rule for printing the content of the inventory:
	say "You are carrying: ";
	list the contents of the player, as a sentence, giving inventory information, including all contents;
	say ".".

The print standard inventory rule is not listed in any rulebook.
Carry out taking inventory (this is the new print inventory rule):
	say "You are carrying: ";
	list the contents of the player, as a sentence, giving inventory information, including all contents;
	say ".".

Printing the content of nowhere is an activity.
Rule for printing the content of nowhere:
	say "Nowhere:[line break]";
	let L be the list of the off-stage things;
	repeat with thing running through L:
		say "  [thing][line break]";

Printing the things on the floor is an activity.
Rule for printing the things on the floor:
	let R be the location of the player;
	let L be the list of things in R;
	remove yourself from L;
	remove the list of containers from L;
	remove the list of supporters from L;
	remove the list of doors from L;
	if the number of entries in L is greater than 0:
		say "There is [L with indefinite articles] on the floor.";

After printing the name of something (called target) while
printing the content of the room
or printing the content of the world
or printing the content of the inventory
or printing the content of nowhere:
	follow the property-aggregation rules for the target.

The property-aggregation rules are an object-based rulebook.
The property-aggregation rulebook has a list of text called the tagline.

[At the moment, we only support "open/unlocked", "closed/unlocked" and "closed/locked" for doors and containers.]
[A first property-aggregation rule for an openable open thing (this is the mention open openables rule):
	add "open" to the tagline.

A property-aggregation rule for an openable closed thing (this is the mention closed openables rule):
	add "closed" to the tagline.

A property-aggregation rule for an lockable unlocked thing (this is the mention unlocked lockable rule):
	add "unlocked" to the tagline.

A property-aggregation rule for an lockable locked thing (this is the mention locked lockable rule):
	add "locked" to the tagline.]

A first property-aggregation rule for an openable lockable open unlocked thing (this is the mention open openables rule):
	add "open" to the tagline.

A property-aggregation rule for an openable lockable closed unlocked thing (this is the mention closed openables rule):
	add "closed" to the tagline.

A property-aggregation rule for an openable lockable closed locked thing (this is the mention locked openables rule):
	add "locked" to the tagline.

A property-aggregation rule for a lockable thing (called the lockable thing) (this is the mention matching key of lockable rule):
	let X be the matching key of the lockable thing;
	if X is not nothing:
		add "match [X]" to the tagline.

A property-aggregation rule for an edible off-stage thing (this is the mention eaten edible rule):
	add "eaten" to the tagline.

The last property-aggregation rule (this is the print aggregated properties rule):
	if the number of entries in the tagline is greater than 0:
		say " ([tagline])";
		rule succeeds;
	rule fails;

The objective part 0 is some text that varies. The objective part 0 is "Welcome to TextWorld! Here is how to play! First step, doublecheck that the birch non-euclidean hatch is shut. Then, try to venture north. Once you get through with that, pick up the iron non-euclidea".
The objective part 1 is some text that varies. The objective part 1 is "n latchkey from the brand new case inside the gloomy basement. After stealing the iron non-euclidean latchkey, try to take a trip south. After that, ensure that the birch non-euclidean hatch is open. ".
The objective part 2 is some text that varies. The objective part 2 is "And then, move south. After that, make sure that the birch non-euclidean hatch is shut. After that, doublecheck that the birch non-euclidean hatch is locked with the iron non-euclidean latchkey. Once ".
The objective part 3 is some text that varies. The objective part 3 is "that's all handled, you can stop!".

An objective is some text that varies. The objective is "[objective part 0][objective part 1][objective part 2][objective part 3]".
Printing the objective is an action applying to nothing.
Carry out printing the objective:
	say "[objective]".

Understand "goal" as printing the objective.

The taking action has an object called previous locale (matched as "from").

Setting action variables for taking:
	now previous locale is the holder of the noun.

Report taking something from the location:
	say "You pick up [the noun] from the ground." instead.

Report taking something:
	say "You take [the noun] from [the previous locale]." instead.

Report dropping something:
	say "You drop [the noun] on the ground." instead.

The print state option is a truth state that varies.
The print state option is usually false.

Turning on the print state option is an action applying to nothing.
Carry out turning on the print state option:
	Now the print state option is true.

Turning off the print state option is an action applying to nothing.
Carry out turning off the print state option:
	Now the print state option is false.

Printing the state is an activity.
Rule for printing the state:
	let R be the location of the player;
	say "Room: [line break] [the internal name of R][line break]";
	[say "[line break]";
	carry out the printing the content of the room activity;]
	say "[line break]";
	carry out the printing the content of the world activity;
	say "[line break]";
	carry out the printing the content of the inventory activity;
	say "[line break]";
	carry out the printing the content of nowhere activity;
	say "[line break]".

Printing the entire state is an action applying to nothing.
Carry out printing the entire state:
	say "-=STATE START=-[line break]";
	carry out the printing the state activity;
	say "[line break]Score:[line break] [score]/[maximum score][line break]";
	say "[line break]Objective:[line break] [objective][line break]";
	say "[line break]Inventory description:[line break]";
	say "  You are carrying: [a list of things carried by the player].[line break]";
	say "[line break]Room description:[line break]";
	try looking;
	say "[line break]-=STATE STOP=-";

Every turn:
	if extra description command option is true:
		say "<description>";
		try looking;
		say "</description>";
	if extra inventory command option is true:
		say "<inventory>";
		try taking inventory;
		say "</inventory>";
	if extra score command option is true:
		say "<score>[line break][score][line break]</score>";
	if extra score command option is true:
		say "<moves>[line break][turn count][line break]</moves>";
	if print state option is true:
		try printing the entire state;

When play ends:
	if print state option is true:
		try printing the entire state;

After looking:
	carry out the printing the things on the floor activity.

Understand "print_state" as printing the entire state.
Understand "enable print state option" as turning on the print state option.
Understand "disable print state option" as turning off the print state option.

Before going through a closed door (called the blocking door):
	say "You have to open the [blocking door] first.";
	stop.

Before opening a locked door (called the locked door):
	let X be the matching key of the locked door;
	if X is nothing:
		say "The [locked door] is welded shut.";
	otherwise:
		say "You have to unlock the [locked door] with the [X] first.";
	stop.

Before opening a locked container (called the locked container):
	let X be the matching key of the locked container;
	if X is nothing:
		say "The [locked container] is welded shut.";
	otherwise:
		say "You have to unlock the [locked container] with the [X] first.";
	stop.

Displaying help message is an action applying to nothing.
Carry out displaying help message:
	say "[fixed letter spacing]Available commands:[line break]";
	say "  look:                describe the current room[line break]";
	say "  goal:                print the goal of this game[line break]";
	say "  inventory:           print player's inventory[line break]";
	say "  go <dir>:            move the player north, east, south or west[line break]";
	say "  examine ...:         examine something more closely[line break]";
	say "  eat ...:             eat edible food[line break]";
	say "  open ...:            open a door or a container[line break]";
	say "  close ...:           close a door or a container[line break]";
	say "  drop ...:            drop an object on the floor[line break]";
	say "  take ...:            take an object that is on the floor[line break]";
	say "  put ... on ...:      place an object on a supporter[line break]";
	say "  take ... from ...:   take an object from a container or a supporter[line break]";
	say "  insert ... into ...: place an object into a container[line break]";
	say "  lock ... with ...:   lock a door or a container with a key[line break]";
	say "  unlock ... with ...: unlock a door or a container with a key[line break]";

Understand "help" as displaying help message.

Taking all is an action applying to nothing.
Check taking all:
	say "You have to be more specific!";
	rule fails.

Understand "take all" as taking all.
Understand "get all" as taking all.
Understand "pick up all" as taking all.

Understand "take each" as taking all.
Understand "get each" as taking all.
Understand "pick up each" as taking all.

Understand "take everything" as taking all.
Understand "get everything" as taking all.
Understand "pick up everything" as taking all.

The extra description command option is a truth state that varies.
The extra description command option is usually false.

Turning on the extra description command option is an action applying to nothing.
Carry out turning on the extra description command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra description command option is true.

Understand "tw-extra-infos description" as turning on the extra description command option.

The extra inventory command option is a truth state that varies.
The extra inventory command option is usually false.

Turning on the extra inventory command option is an action applying to nothing.
Carry out turning on the extra inventory command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra inventory command option is true.

Understand "tw-extra-infos inventory" as turning on the extra inventory command option.

The extra score command option is a truth state that varies.
The extra score command option is usually false.

Turning on the extra score command option is an action applying to nothing.
Carry out turning on the extra score command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra score command option is true.

Understand "tw-extra-infos score" as turning on the extra score command option.

The extra moves command option is a truth state that varies.
The extra moves command option is usually false.

Turning on the extra moves command option is an action applying to nothing.
Carry out turning on the extra moves command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra moves command option is true.

Understand "tw-extra-infos moves" as turning on the extra moves command option.

To trace the actions:
	(- trace_actions = 1; -).

Tracing the actions is an action applying to nothing.
Carry out tracing the actions:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	trace the actions;

Understand "tw-trace-actions" as tracing the actions.

The restrict commands option is a truth state that varies.
The restrict commands option is usually false.

Turning on the restrict commands option is an action applying to nothing.
Carry out turning on the restrict commands option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the restrict commands option is true.

Understand "restrict commands" as turning on the restrict commands option.

The taking allowed flag is a truth state that varies.
The taking allowed flag is usually false.

Before removing something from something:
	now the taking allowed flag is true.

After removing something from something:
	now the taking allowed flag is false.

Before taking a thing (called the object) when the object is on a supporter (called the supporter):
	if the restrict commands option is true and taking allowed flag is false:
		say "Can't see any [object] on the floor! Try taking the [object] from the [supporter] instead.";
		rule fails.

Before of taking a thing (called the object) when the object is in a container (called the container):
	if the restrict commands option is true and taking allowed flag is false:
		say "Can't see any [object] on the floor! Try taking the [object] from the [container] instead.";
		rule fails.

Understand "take [something]" as removing it from.

Rule for supplying a missing second noun while removing:
	if restrict commands option is false and noun is on a supporter (called the supporter):
		now the second noun is the supporter;
	else if restrict commands option is false and noun is in a container (called the container):
		now the second noun is the container;
	else:
		try taking the noun;
		say ""; [Needed to avoid printing a default message.]

The version number is always 1.

Reporting the version number is an action applying to nothing.
Carry out reporting the version number:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	say "[version number]".

Understand "tw-print version" as reporting the version number.

Reporting max score is an action applying to nothing.
Carry out reporting max score:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	if maximum score is -32768:
		say "infinity";
	else:
		say "[maximum score]".

Understand "tw-print max_score" as reporting max score.

To print id of (something - thing):
	(- print {something}, "^"; -).

Printing the id of player is an action applying to nothing.
Carry out printing the id of player:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	print id of player.

Printing the id of EndOfObject is an action applying to nothing.
Carry out printing the id of EndOfObject:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	print id of EndOfObject.

Understand "tw-print player id" as printing the id of player.
Understand "tw-print EndOfObject id" as printing the id of EndOfObject.

There is a EndOfObject.

