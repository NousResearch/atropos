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


The r_1 and the r_0 and the r_2 and the r_3 and the r_4 and the r_5 are rooms.

Understand "serious office" as r_1.
The internal name of r_1 is "serious office".
The printed name of r_1 is "-= Serious Office =-".
The serious office part 0 is some text that varies. The serious office part 0 is "You've moved into a serious room. Your mind races to think of what kind of room would be serious. And then it hits you. Of course. You're in the serious office.



 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The serious office part 1 is some text that varies. The serious office part 1 is " wooden hatch leading south. You need an unguarded exit? You should try going west.".
The description of r_1 is "[serious office part 0][serious office part 1]".

The r_0 is mapped west of r_1.
south of r_1 and north of r_2 is a door called d_1.
Understand "stuffed garage" as r_0.
The internal name of r_0 is "stuffed garage".
The printed name of r_0 is "-= Stuffed Garage =-".
The stuffed garage part 0 is some text that varies. The stuffed garage part 0 is "Wow! You're in a stuffed garage.

 You can see a brand new suitcase. Huh, weird.[if c_1 is open and there is something in the c_1] The brand new suitcase contains [a list of things in the c_1]. Now why would someone leave that there?[end if]".
The stuffed garage part 1 is some text that varies. The stuffed garage part 1 is "[if c_1 is open and the c_1 contains nothing] The suitcase is empty, what a horrible day![end if]".
The stuffed garage part 2 is some text that varies. The stuffed garage part 2 is "

You don't like doors? Why not try going east, that entranceway is unguarded.".
The description of r_0 is "[stuffed garage part 0][stuffed garage part 1][stuffed garage part 2]".

The r_1 is mapped east of r_0.
Understand "exciting bar" as r_2.
The internal name of r_2 is "exciting bar".
The printed name of r_2 is "-= Exciting Bar =-".
The exciting bar part 0 is some text that varies. The exciting bar part 0 is "Welcome to the exciting bar. I guess you better just go and list everything you see here.

 You can make out [if c_2 is locked]a locked[else if c_2 is open]an opened[otherwise]a closed[end if]".
The exciting bar part 1 is some text that varies. The exciting bar part 1 is " chartreuse locker.[if c_2 is open and there is something in the c_2] The chartreuse locker contains [a list of things in the c_2]. There's something strange about this being here, but you can't put your finger on it.[end if]".
The exciting bar part 2 is some text that varies. The exciting bar part 2 is "[if c_2 is open and the c_2 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The exciting bar part 3 is some text that varies. The exciting bar part 3 is "

 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The exciting bar part 4 is some text that varies. The exciting bar part 4 is " wooden hatch leading north. There is [if d_0 is open]an open[otherwise]a closed[end if]".
The exciting bar part 5 is some text that varies. The exciting bar part 5 is " stone passageway leading east. You don't like doors? Why not try going west, that entranceway is unguarded.".
The description of r_2 is "[exciting bar part 0][exciting bar part 1][exciting bar part 2][exciting bar part 3][exciting bar part 4][exciting bar part 5]".

The r_3 is mapped west of r_2.
north of r_2 and south of r_1 is a door called d_1.
east of r_2 and west of r_4 is a door called d_0.
Understand "sweaty cookhouse" as r_3.
The internal name of r_3 is "sweaty cookhouse".
The printed name of r_3 is "-= Sweaty Cookhouse =-".
The sweaty cookhouse part 0 is some text that varies. The sweaty cookhouse part 0 is "Well, here we are in the sweaty cookhouse. You start to take note of what's in the room.

 You can make out a formless safe. You wonder idly who left that here.[if c_0 is open and there is something in the c_0] The sturdy formless safe contains [a list of things in the c_0].[end if]".
The sweaty cookhouse part 1 is some text that varies. The sweaty cookhouse part 1 is "[if c_0 is open and the c_0 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The sweaty cookhouse part 2 is some text that varies. The sweaty cookhouse part 2 is " You smell an intriguing smell, and follow it to a yellow locker.[if c_3 is open and there is something in the c_3] The yellow locker contains [a list of things in the c_3].[end if]".
The sweaty cookhouse part 3 is some text that varies. The sweaty cookhouse part 3 is "[if c_3 is open and the c_3 contains nothing] The locker is empty! What a waste of a day![end if]".
The sweaty cookhouse part 4 is some text that varies. The sweaty cookhouse part 4 is "

There is an exit to the east. Don't worry, it is unblocked.".
The description of r_3 is "[sweaty cookhouse part 0][sweaty cookhouse part 1][sweaty cookhouse part 2][sweaty cookhouse part 3][sweaty cookhouse part 4]".

The r_2 is mapped east of r_3.
Understand "clean washroom" as r_4.
The internal name of r_4 is "clean washroom".
The printed name of r_4 is "-= Clean Washroom =-".
The clean washroom part 0 is some text that varies. The clean washroom part 0 is "You arrive in a washroom. A clean one.



 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The clean washroom part 1 is some text that varies. The clean washroom part 1 is " stone passageway leading west. There is an exit to the north. Don't worry, it is unblocked.".
The description of r_4 is "[clean washroom part 0][clean washroom part 1]".

west of r_4 and east of r_2 is a door called d_0.
The r_5 is mapped north of r_4.
Understand "steamy laundry place" as r_5.
The internal name of r_5 is "steamy laundry place".
The printed name of r_5 is "-= Steamy Laundry Place =-".
The steamy laundry place part 0 is some text that varies. The steamy laundry place part 0 is "You've entered a steamy laundry place.



There is an unblocked exit to the south.".
The description of r_5 is "[steamy laundry place part 0]".

The r_4 is mapped south of r_5.

The c_0 and the c_1 and the c_2 and the c_3 are containers.
The c_0 and the c_1 and the c_2 and the c_3 are privately-named.
The d_1 and the d_0 are doors.
The d_1 and the d_0 are privately-named.
The f_0 are foods.
The f_0 are privately-named.
The k_1 and the k_2 and the k_3 are keys.
The k_1 and the k_2 and the k_3 are privately-named.
The o_0 and the o_1 are object-likes.
The o_0 and the o_1 are privately-named.
The r_1 and the r_0 and the r_2 and the r_3 and the r_4 and the r_5 are rooms.
The r_1 and the r_0 and the r_2 and the r_3 and the r_4 and the r_5 are privately-named.

The description of d_1 is "it is what it is, a wooden hatch [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_1 is "wooden hatch".
Understand "wooden hatch" as d_1.
Understand "wooden" as d_1.
Understand "hatch" as d_1.
The d_1 is closed.
The description of d_0 is "it's an imposing passageway [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_0 is "stone passageway".
Understand "stone passageway" as d_0.
Understand "stone" as d_0.
Understand "passageway" as d_0.
The d_0 is open.
The description of c_0 is "The sturdy formless safe looks strong, and impossible to crack. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_0 is "sturdy formless safe".
Understand "sturdy formless safe" as c_0.
Understand "sturdy" as c_0.
Understand "formless" as c_0.
Understand "safe" as c_0.
The c_0 is in r_3.
The c_0 is open.
The description of c_1 is "The brand new suitcase looks strong, and impossible to destroy. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_1 is "brand new suitcase".
Understand "brand new suitcase" as c_1.
Understand "brand" as c_1.
Understand "new" as c_1.
Understand "suitcase" as c_1.
The c_1 is in r_0.
The c_1 is locked.
The description of c_2 is "The chartreuse locker looks strong, and impossible to destroy. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_2 is "chartreuse locker".
Understand "chartreuse locker" as c_2.
Understand "chartreuse" as c_2.
Understand "locker" as c_2.
The c_2 is in r_2.
The c_2 is locked.
The description of c_3 is "The yellow locker looks strong, and impossible to crack. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_3 is "yellow locker".
Understand "yellow locker" as c_3.
Understand "yellow" as c_3.
Understand "locker" as c_3.
The c_3 is in r_3.
The c_3 is closed.
The description of f_0 is "You couldn't pay me to eat that massive thing.".
The printed name of f_0 is "massive peanut".
Understand "massive peanut" as f_0.
Understand "massive" as f_0.
Understand "peanut" as f_0.
The f_0 is in the c_0.
The description of k_1 is "The metal of the chartreuse latchkey is hammered.".
The printed name of k_1 is "chartreuse latchkey".
Understand "chartreuse latchkey" as k_1.
Understand "chartreuse" as k_1.
Understand "latchkey" as k_1.
The player carries the k_1.
The matching key of the c_2 is the k_1.
The description of k_2 is "The yellow key looks useful".
The printed name of k_2 is "yellow key".
Understand "yellow key" as k_2.
Understand "yellow" as k_2.
Understand "key" as k_2.
The player carries the k_2.
The matching key of the c_3 is the k_2.
The description of k_3 is "The iron formless latchkey is weighty.".
The printed name of k_3 is "iron formless latchkey".
Understand "iron formless latchkey" as k_3.
Understand "iron" as k_3.
Understand "formless" as k_3.
Understand "latchkey" as k_3.
The k_3 is in the c_0.
The matching key of the c_0 is the k_3.
The description of o_0 is "The ornate bowl appears to be out of place here".
The printed name of o_0 is "ornate bowl".
Understand "ornate bowl" as o_0.
Understand "ornate" as o_0.
Understand "bowl" as o_0.
The player carries the o_0.
The description of o_1 is "The tiny nest of ticks is unremarkable.".
The printed name of o_1 is "tiny nest of ticks".
Understand "tiny nest of ticks" as o_1.
Understand "tiny" as o_1.
Understand "nest" as o_1.
Understand "ticks" as o_1.
The o_1 is in the c_0.


The player is in r_3.

The quest0 completed is a truth state that varies.
The quest0 completed is usually false.

Test quest0_0 with "take massive peanut from sturdy formless safe"

Every turn:
	if quest0 completed is true:
		do nothing;
	else if The player is in r_3 and The c_0 is in r_3 and The c_0 is open and The player carries the f_0:
		increase the score by 1; [Quest completed]
		if 1 is 1 [always true]:
			Now the quest0 completed is true;

The quest1 completed is a truth state that varies.
The quest1 completed is usually false.

Test quest1_0 with "take massive peanut from sturdy formless safe / go east / open wooden hatch / go north / eat massive peanut"

Every turn:
	if quest1 completed is true:
		do nothing;
	else if The f_0 is nowhere:
		increase the score by 1; [Quest completed]
		if 1 is 1 [always true]:
			Now the quest1 completed is true;

Use scoring. The maximum score is 2.
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
	if quest0 completed is true and quest1 completed is true:
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

The objective part 0 is some text that varies. The objective part 0 is "Hey, thanks for coming over to the TextWorld today, there is something I need you to do for me. Your first objective is to take the massive peanut from the sturdy formless safe. After taking the massi".
The objective part 1 is some text that varies. The objective part 1 is "ve peanut, eat the massive peanut. Alright, thanks!".

An objective is some text that varies. The objective is "[objective part 0][objective part 1]".
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

